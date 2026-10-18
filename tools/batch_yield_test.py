#!/usr/bin/env python3
"""BYIELD beside a decoding slot (cooperative preemption, #656): a long prompt read gives way to a short one at a chunk
boundary while another slot keeps decoding.  Greedy; every request's tokens must equal its solo tokens.

  1. solo references (GEN): A (short), D (long, --long-tokens of filler), E (short)
  2. A in slot 0, decoding; D into slot 1: once A's tokens arrive between D's chunks, BYIELD 1 - D's read stops at the
     next chunk boundary (YIELDED 1 <tokens read> + DONE cancel + BADM 1 0), the part read waits in slot 1
  3. E into slot 2 beside A, then D sent again into slot 1: it goes on (from its slot, or read again where a slot cannot
     be copied back yet) and decodes beside A and E
  4. A, D and E each equal their solo tokens; A's decode rate during D's reads is printed

  python3 tools/batch_yield_test.py --exe build/strata --config strata-<model>.json --long-tokens 8000
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

FILLER = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open risks, and "
          "asked the staff to report back next quarter. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--long-tokens", type=int, default=8000, help="about this many tokens of filler in D's prompt")
    ap.add_argument("--max-new", type=int, default=64, help="tokens for D and E")
    ap.add_argument("--max-new-a", type=int, default=1024, help="tokens for A (it decodes through both reads of D)")
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--max-context 65536"')
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    per = max(1, len(tok.encode(FILLER % 100)))
    A = chat("Describe the life cycle of a star like the Sun, in great detail.")
    D = chat("".join(FILLER % i for i in range(a.long_tokens // per + 1)) + "\n\nSummarize the agenda in three sentences.")
    E = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    env = {"STRATA_IQ_MT_MIN": a.mt_min} if a.mt_min else {}
    eng = Engine(a.exe, cfg, 3, env, a.extra.split())
    out = eng.lines()
    ids = lambda v: ",".join(map(str, v))

    def solo(P, m):
        eng.send(f"GEN {m} {ids(P)}")
        got = []
        for l in out:
            if l.startswith("T "):
                got.append(int(l.split()[1]))
            elif l.startswith(("DONE", "ERR")):
                if l.startswith("ERR"):
                    raise SystemExit("engine: " + l)
                return got
    sA, sD, sE = solo(A, a.max_new_a), solo(D, a.max_new), solo(E, a.max_new)
    print(f"prompts: A {len(A)}, D {len(D)}, E {len(E)} tokens; solo A {len(sA)}, D {len(sD)}, E {len(sE)}", flush=True)

    got = {0: [], 1: [], 2: []}
    done = set()

    def admit(s, P, m, yield_after=0):
        """BGEN s; the slots' BT lines are kept.  yield_after > 0: BYIELD s once that many slot tokens arrived during the
        read.  Returns (BADM flag, YIELDED line or None, slot tokens during the read, seconds)."""
        eng.send(f"BGEN {s} {m} {ids(P)}")
        t0, beside, yielded, sent = time.time(), 0, None, False
        for l in out:
            if l.startswith("T "):
                got[s].append(int(l.split()[1]))
            elif l.startswith("BT "):
                _, b, y = l.split()
                got[int(b)].append(int(y))
                beside += 1
                if yield_after and not sent and beside >= yield_after:
                    eng.send(f"BYIELD {s}")
                    sent = True
            elif l.startswith("BDONE "):
                done.add(int(l.split()[1]))
            elif l.startswith("YIELDED "):
                yielded = l
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
            elif l.startswith(f"BADM {s} "):
                if l.split()[2] == "0":
                    done.add(s)
                return l.split()[2] == "1", yielded, beside, time.time() - t0
        raise SystemExit(f"the engine ended (see {eng.log_path})")

    ok = True
    cont, _, _, _ = admit(0, A, a.max_new_a)
    if not cont:
        print("A did not go on in its slot")
        return 2
    _, yielded, beside1, dt1 = admit(1, D, a.max_new, yield_after=2)
    print(f"D's first read: {dt1:.1f}s, A decoded {beside1} tokens meanwhile ({beside1 / max(dt1, 1e-9):.1f} tok/s); "
          f"{yielded or 'no YIELDED'}", flush=True)
    if yielded is None or got[1]:
        print("FAIL: D did not give way (or produced tokens before it did)")
        ok = False
    got[1] = []
    admit(2, E, a.max_new)
    d0 = len(got[0])
    _, _, beside2, dt2 = admit(1, D, a.max_new)
    print(f"D sent again: {dt2:.1f}s to its first token, A and E decoded {beside2} tokens meanwhile "
          f"(A {len(got[0]) - d0}: {(len(got[0]) - d0) / max(dt2, 1e-9):.1f} tok/s)", flush=True)
    if 0 in done and len(got[0]) < len(sA):
        print("FAIL: A ended early")
        ok = False
    while len(done) < 3:
        l = next(out)
        if l.startswith("BT "):
            _, b, y = l.split()
            got[int(b)].append(int(y))
        elif l.startswith("BDONE "):
            done.add(int(l.split()[1]))
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
    for s, ref, name in ((0, sA, "A (decoding through both reads)"), (1, sD, "D (gave way, then sent again)"),
                         (2, sE, "E (admitted while D waited)")):
        same = got[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(ref))) if got[s][k] != ref[k]), None)
        print(f"slot {s} {name}: {len(got[s])} tokens, solo {len(ref)}: {'IDENTICAL' if same else f'DIFFERS at {d}'}",
              flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
