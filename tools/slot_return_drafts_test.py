#!/usr/bin/env python3
"""MTP drafts after a lone batch slot goes back to the solo path: through the --serve protocol, one engine with
--batch 2 and MTP (--spec), greedy.  A slot runs no drafter, so the solo path's drafter K/V below the slot's tokens
hold the conversation it read before; this measures what that costs against a conversation that never left it.

Per round:
  slot:  GEN X (another conversation fills the solo window and its drafter), BGEN 0 P + BGEN 1 short, slot 1 ends,
         BSTOP 0 after >= 12 tokens (out0), then GEN P+out0 N: the slot's conversation back on the solo path.
  ref:   GEN X again, GEN P len(out0) (solo, never in a slot), GEN P+out0 N (its live continuation).
Prints drafts accepted / offered and the decode rate of the N-token continuation for both.  Checks: the tokens are
the same, and the acceptance after the slot is within --tolerance of the reference's (all rounds together).

  python3 tools/slot_return_drafts_test.py --exe <absolute engine path> --config strata-<model>.json [--rounds 3]

Exit 0 when the checks pass, 1 otherwise.  STRATA_IQ_MT_MIN=1 (as batch_test.py) for exact tokens.
"""
import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p-tokens", type=int, default=1500)
    ap.add_argument("--x-tokens", type=int, default=3000)
    ap.add_argument("--extra", default="", help="more engine arguments in one string")
    ap.add_argument("--tolerance", type=float, default=0.05, help="the acceptance after the slot may be this much lower")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    enc = lambda t: tok.encode(t, parse_special=True)
    nothink = lambda q: enc(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def filler(para, n):
        k = n // max(1, len(enc(para % 100))) + 1
        return "".join(para % i for i in range(k))

    xpara = ("Shipment %d of copper wire left the northern warehouse on a grey morning, logged by the night clerk "
             "with its weight, its seal number and the name of the driver. ")
    ppara = ("In chapter %d the old lighthouse keeper climbed the spiral stairs, trimmed the wick, wound the clockwork "
             "and watched the fishing boats come home through the fog. ")
    X = nothink(filler(xpara, a.x_tokens) + "\n\nSummarise the shipping log in detail.")
    P = nothink(filler(ppara, a.p_tokens) + "\n\nContinue this story about the lighthouse keeper for several "
                "paragraphs, then describe the message he finds in a bottle.")
    S = nothink("Name three primary colours.")
    env = dict(cfg.get("env") or {}, STRATA_IQ_MT_MIN="1")
    os.environ["BATCH_TEST_LOG"] = os.path.join(tempfile.gettempdir(), "slot_return_drafts_test.log")
    eng = BT.Engine(os.path.abspath(a.exe), cfg, 2, env, a.extra.split())
    out = eng.lines()
    slot = {0: [], 1: []}
    bdone = {}

    def line():
        l = next(out, None)
        if l is None or l.startswith("ERR"):
            raise SystemExit(f"engine: {l} (log {eng.log_path})")
        if l.startswith("BT "):
            _, s, y = l.split()
            slot[int(s)].append(int(y))
        elif l.startswith("BDONE "):
            bdone[int(l.split()[1])] = l
        return l

    def gen(ids, n):
        eng.send(f"GEN {n} " + ",".join(map(str, ids)))
        got = []
        while True:
            l = line()
            if l.startswith("T "):
                got.append(int(l.split()[1]))
            elif l.startswith("DONE"):
                f = l.split()
                return got, {"gen": int(f[1]), "decode_ms": float(f[4]), "ok": int(f[6]), "drafts": int(f[7]),
                             "reused": int(f[8])}

    def bgen(b, ids, n):
        slot[b] = []
        bdone.pop(b, None)
        eng.send(f"BGEN {b} {n} " + ",".join(map(str, ids)))
        while True:
            l = line()
            if l.startswith("T "):
                slot[b].append(int(l.split()[1]))
            elif l.startswith("BADM "):
                if l.split()[2] == "0":
                    bdone[b] = l
                return

    def until(c):
        while not c():
            line()

    rate = lambda d: d["gen"] / d["decode_ms"] * 1000.0
    res = []
    for r in range(a.rounds):
        gen(X, 16)
        bgen(0, P, 64)
        bgen(1, S, 4)
        until(lambda: 1 in bdone and len(slot[0]) >= 12)
        eng.send("BSTOP 0")
        until(lambda: 0 in bdone)
        out0 = list(slot[0])
        rest_s, ds = gen(P + out0, a.n)
        gen(X, 16)
        o0, _ = gen(P, len(out0))
        rest_r, dr = gen(P + o0, a.n)
        same = o0 == out0 and rest_r == rest_s
        print(f"round {r + 1}: after slot   drafts {ds['ok']}/{ds['drafts']} ({ds['ok'] / max(1, ds['drafts']):.3f}), "
              f"{rate(ds):.1f} tok/s, reused {ds['reused']}", flush=True)
        print(f"round {r + 1}: never in slot drafts {dr['ok']}/{dr['drafts']} ({dr['ok'] / max(1, dr['drafts']):.3f}), "
              f"{rate(dr):.1f} tok/s, reused {dr['reused']}; tokens {'IDENTICAL' if same else 'DIFFER'}", flush=True)
        res.append((ds, dr, same))
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    tot = lambda k, i: sum(x[i][k] for x in res)
    acc = lambda i: tot("ok", i) / max(1, tot("drafts", i))
    print(f"total: after slot {tot('ok', 0)}/{tot('drafts', 0)} = {acc(0):.3f}; "
          f"never in slot {tot('ok', 1)}/{tot('drafts', 1)} = {acc(1):.3f}")
    ok_tokens = all(x[2] for x in res)
    ok_acc = acc(0) >= acc(1) - a.tolerance
    print(f"{'PASS' if ok_tokens else 'FAIL'} tokens: identical in {sum(x[2] for x in res)}/{len(res)} rounds")
    print(f"{'PASS' if ok_acc else 'FAIL'} acceptance after the slot {acc(0):.3f} vs {acc(1):.3f} (tolerance {a.tolerance})")
    return 0 if ok_tokens and ok_acc else 1


if __name__ == "__main__":
    sys.exit(main())
