#!/usr/bin/env python3
"""Batch slot lifecycle test: through the --serve protocol, an engine with --batch 2, greedy.

  1. back to solo: a request decoding in slot 0 next to a short one in slot 1 is left alone, BSTOPped, and continued
     with GEN <prompt + its tokens> (what serve/server.py does, STRATA_PARALLEL_SOLO): the engine reuses the slot's
     conversation (DONE's reused field = everything the slot holds; the log says "slot 0 gave back") and the tokens
     after it equal the same request decoded alone.
  2. resume in the slot: the next turn of a conversation a slot holds (--long-tokens of filler first) (its prompt + its whole reply + a new user turn)
     is admitted into that slot again and reads only the new turn (reused = what the slot holds); its tokens equal a
     reference conversation that stayed on the solo path.
  3. the same with the reply's thinking dropped (the client renders the previous reply without it): the slot's turn
     checkpoint (at the turn's last <|im_start|>) is reused, and the tokens equal the solo reference's.
  4. BSTOP on slot 1 leaves slot 0's tokens equal to its solo tokens.

The references are decoded first by an engine without --batch (a conversation that never left the solo path), then
the engine with --batch 2 starts (one at a time).

  python3 tools/slot_lifecycle_test.py --exe build-metal/strata --config strata-<model>.json [--extra "..."]

Exit 0 when every check passes, 1 otherwise.  STRATA_IQ_MT_MIN=1 (as batch_test.py) for exact tokens.
"""
import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

TURN = 248045   # <|im_start|> (the engine's --turn-token default)


class Proto:
    """The engine's stdout as events: solo tokens (T), slot tokens (BT), DONE / BADM / BDONE lines."""

    def __init__(self, eng):
        self.eng, self.out = eng, eng.lines()
        self.slot = {0: [], 1: []}
        self.bdone = {}

    def send(self, line):
        self.eng.send(line)

    def _line(self):
        line = next(self.out, None)
        if line is None:
            raise SystemExit(f"the engine ended - see {self.eng.log_path}")
        if line.startswith("ERR"):
            raise SystemExit(f"engine: {line}")
        if line.startswith("BT "):
            _, s, y = line.split()
            self.slot[int(s)].append(int(y))
        elif line.startswith("BDONE "):
            self.bdone[int(line.split()[1])] = line
        return line

    def until(self, cond):
        while not cond():
            self._line()

    def gen(self, ids, max_new):
        """GEN: (tokens, reused)"""
        self.send(f"GEN {max_new} " + ",".join(map(str, ids)))
        got = []
        while True:
            line = self._line()
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return got, int(line.split()[8])

    def bgen(self, b, ids, max_new):
        """BGEN into slot b: its first token, then it decodes in the batch windows; returns reused"""
        self.slot[b], reused = [], None
        self.bdone.pop(b, None)
        self.send(f"BGEN {b} {max_new} " + ",".join(map(str, ids)))
        while True:
            line = self._line()
            if line.startswith("T "):
                self.slot[b].append(int(line.split()[1]))
            elif line.startswith("DONE"):
                reused = int(line.split()[8])
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    self.bdone[b] = line
                return reused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--extra", default="", help="more engine arguments in one string")
    ap.add_argument("--long-tokens", type=int, default=4096,
                    help="conversation 2 starts with about this many tokens of filler text (its K/V spans many "
                         "indexer blocks, so the resumed slot's attention selects among them)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    enc = lambda t: tok.encode(t, parse_special=True)
    nothink = lambda q: enc(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    think = lambda q: enc(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    Q, M = BT.QUESTIONS, a.max_new
    env = dict(cfg.get("env") or {}, STRATA_IQ_MT_MIN="1")
    exe = os.path.abspath(a.exe)   # (the engine runs in the config's cwd)
    P0, P1 = nothink(Q[0]), nothink(Q[1])
    para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open risks, "
            "and asked the staff to report back next quarter. ")
    filler = "".join(para % i for i in range(a.long_tokens // max(1, len(enc(para % 100))) + 1)) if a.long_tokens else ""
    A, B = nothink(f"{filler}\n\n{Q[2]}" if filler else Q[2]), think(Q[3])
    C, D = nothink(Q[4]), nothink(Q[5])
    follow = enc("<|im_end|>\n<|im_start|>user\nNow summarise that in three sentences.<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    turn = max(i for i, t in enumerate(B) if t == TURN)     # the turn checkpoint: B up to its last <|im_start|>
    B2 = B[:turn] + enc("<|im_start|>assistant\nHere is a short answer.<|im_end|>\n<|im_start|>user\nAnd in one "
                        "sentence?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    # the references: the same requests and conversations on an engine without batch slots (one engine at a time)
    os.environ["BATCH_TEST_LOG"] = os.path.join(tempfile.gettempdir(), "slot_lifecycle_test_ref.log")
    pr = Proto(BT.Engine(exe, cfg, 0, env, a.extra.split()))
    ref0, _ = pr.gen(P0, M)
    refA, _ = pr.gen(A, M)
    refA2, _ = pr.gen(A + refA + follow, M)
    refB, _ = pr.gen(B, M)
    refB2, _ = pr.gen(B2, M)
    refC, _ = pr.gen(C, M)
    pr.send("QUIT")
    pr.eng.p.wait(timeout=180)

    log_path = os.path.join(tempfile.gettempdir(), "slot_lifecycle_test_engine.log")
    os.environ["BATCH_TEST_LOG"] = log_path
    pr = Proto(BT.Engine(exe, cfg, 2, env, a.extra.split()))
    log_at = lambda: os.path.getsize(log_path)
    log_since = lambda at: Path(log_path).read_text(errors="replace")[at:]
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    def same(x, y):
        d = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), None if len(x) == len(y) else min(len(x), len(y)))
        return "IDENTICAL" if d is None else f"DIFFERS at {d}"

    # ---- 1. a lone slot back to the solo path
    pr.bgen(0, P0, M)
    pr.bgen(1, P1, 4)
    pr.until(lambda: 1 in pr.bdone and len(pr.slot[0]) >= 12)
    pr.send("BSTOP 0")
    pr.until(lambda: 0 in pr.bdone)
    out0 = list(pr.slot[0])
    if pr.bdone[0].split()[3] != "cancel":
        raise SystemExit(f"slot 0 ended before its BSTOP ({pr.bdone[0]}): try a larger --max-new")
    at = log_at()
    rest, reused = pr.gen(P0 + out0, M - len(out0))
    held = len(P0) + len(out0) - 1
    check("lone slot back to solo: reuses the slot", reused == held and "slot 0 gave back" in log_since(at),
          f"reused {reused} of {len(P0) + len(out0)} (the slot holds {held})")
    check("lone slot back to solo: tokens", out0 + rest == ref0,
          f"{len(out0)} in the slot + {len(rest)} solo vs {len(ref0)} alone: {same(out0 + rest, ref0)}")

    # ---- 2./3. the next turn of a conversation held in a slot (slot 0: whole reply; slot 1: thinking dropped)
    pr.bgen(0, A, M)
    pr.bgen(1, B, M)
    pr.until(lambda: 0 in pr.bdone and 1 in pr.bdone)
    outA, outB = list(pr.slot[0]), list(pr.slot[1])
    check("resume in slot: turn 1 tokens", outA == refA and outB == refB, f"A {same(outA, refA)}, B {same(outB, refB)}")
    A2 = A + outA + follow
    at = log_at()
    rA = pr.bgen(0, A2, M)
    pr.until(lambda: 0 in pr.bdone)
    gotA2 = list(pr.slot[0])
    check("resume in slot: reads only the new turn", rA == len(A) + len(outA) - 1 and "slot 0 gave back" in log_since(at),
          f"reused {rA} of {len(A2)} (the slot holds {len(A) + len(outA) - 1})")
    check("resume in slot: turn 2 tokens", gotA2 == refA2, f"{len(gotA2)} vs {len(refA2)}: {same(gotA2, refA2)}")
    at = log_at()
    rB = pr.bgen(1, B2, M)
    pr.until(lambda: 1 in pr.bdone)
    gotB2 = list(pr.slot[1])
    check("resume in slot, thinking dropped: the turn checkpoint", rB == turn and "slot 1 gave back" in log_since(at),
          f"reused {rB} of {len(B2)} (the turn checkpoint is at {turn})")
    check("resume in slot, thinking dropped: turn 2 tokens", gotB2 == refB2,
          f"{len(gotB2)} vs {len(refB2)}: {same(gotB2, refB2)}")

    # ---- 4. BSTOP on one slot
    pr.bgen(0, C, M)
    pr.bgen(1, D, M)
    pr.until(lambda: len(pr.slot[1]) >= 8)
    pr.send("BSTOP 1")
    pr.until(lambda: 0 in pr.bdone and 1 in pr.bdone)
    check("BSTOP ends only its slot", pr.slot[0] == refC and pr.bdone[1].split()[3] == "cancel",
          f"slot 1 {pr.bdone[1].split()[3]} after {len(pr.slot[1])}; slot 0 {same(pr.slot[0], refC)}")

    pr.send("QUIT")
    pr.eng.p.wait(timeout=180)
    print("ALL PASS" if all(results) else f"{results.count(False)} FAILED - engine log {log_path}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
