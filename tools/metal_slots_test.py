#!/usr/bin/env python3
"""Batch slots on the Mac, end to end.  Two parts, each one engine start:

  engine  Through the --serve protocol: --batch N at the config's context.  The engine runs the slots that fit in
          free memory and says how many in `INFO ... batch_slots=M`; when M < N a WARNING line names the count and
          the limit that held it.  Checks: INFO batch_slots matches the line; with --expect all, M == N and no
          warning; with --expect fewer, M < N.  Prints the footprint (phys_footprint) 5 s after READY.

  http    Through serve/server.py with "parallel": N (a copy of the config with --exe and --port; never 8080):
          one client alone, then N clients at once, streaming /v1/chat/completions.  Checks: the N streams overlap in
          time (each one's first token comes before another's last), /metrics live.slots shows N slots decoding at
          one moment, /v1/status concurrency.serving == N.  Prints per-stream and total decode rates (tokens / time
          from a stream's first token to its last) for 1 and N clients.

  python3 tools/metal_slots_test.py engine --exe build-metal/strata --config strata-<model>.json --batch 8 \
      [--context 262144] [--expect all|fewer|any]
  python3 tools/metal_slots_test.py http --exe build-metal/strata --config strata-<model>.json --port 8197 \
      [--parallel 2] [--context 262144] [--max-tokens 256]

Exit 0 when every check passes, 1 otherwise.  On a Mac run each part alone (one engine at a time).
"""
import argparse, json, os, re, subprocess, sys, tempfile, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

PROMPTS = ["Write a story about a lighthouse keeper who finds a message in a bottle.",
           "Write a Python function that merges overlapping intervals, with a docstring and tests, then explain it.",
           "List the prime numbers below 500, separated by commas, then explain how you checked them.",
           "Explain how a refrigerator works, step by step, for a twelve-year-old."]

fails = []


def check(name, ok, seen):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {seen}", flush=True)
    if not ok:
        fails.append(name)


def phys_footprint(pid):
    import ctypes
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = ctypes.create_string_buffer(512)
    if lib.proc_pid_rusage(int(pid), 2, buf) != 0:      # RUSAGE_INFO_V2: ri_phys_footprint at byte 72
        return -1
    return int.from_bytes(buf.raw[72:80], "little")


def with_context(args, ctx):
    args = list(args)
    if ctx:
        if "--max-context" in args:
            args[args.index("--max-context") + 1] = str(ctx)
        else:
            args += ["--max-context", str(ctx)]
    return args


def slot_line(log_text):
    """The engine's slot-count line: `--batch N: M batch slot(s) fit ...` (or `--batch is off: ...`)."""
    for l in log_text.splitlines():
        if re.search(r"--batch \d+: \d+ batch slots? fits?", l) or ("--batch is off" in l and "fit" in l):
            return l
    return ""


def run_engine(a):
    cfg = json.loads(Path(a.config).read_text())
    args = [a.exe, "--serve", *with_context(cfg["args"], a.context), "--batch", str(a.batch)]
    log_path = os.path.join(tempfile.gettempdir(), "metal_slots_test_engine.log")
    with open(log_path, "w") as log:
        p = subprocess.Popen(args, cwd=cfg.get("cwd"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, bufsize=1, env=dict(os.environ, **(cfg.get("env") or {})))
    head = []
    for line in p.stdout:
        head.append(line.rstrip("\n"))
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY - see {log_path}")
    time.sleep(5)
    fp = phys_footprint(p.pid)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    info = next((l for l in head if l.startswith("INFO")), "")
    m = re.search(r"batch_slots=(\d+)", info)
    slots = int(m.group(1)) if m else 0
    log_text = Path(log_path).read_text()
    line = slot_line(log_text)
    said = re.search(r"--batch \d+: (\d+) batch slot", line)
    said_n = int(said.group(1)) if said else None
    if "--batch is off" in log_text:                    # fewer than two fit: none run
        said_n = 0
    sizes = [l for l in log_text.splitlines() if "batch slots of" in l or "batch slots fit" in l or "slot room" in l]
    print(f"INFO batch_slots={slots}; engine: {line or '(no slot-count warning)'}")
    for l in sizes:
        print("  " + l)
    print(f"footprint 5 s after READY: {fp / 2 ** 30:.3f} GiB ({fp} B) at --max-context "
          f"{a.context or cfg['args'][cfg['args'].index('--max-context') + 1]}, --batch {a.batch}")
    if a.expect == "all":
        check("all slots asked run", slots == a.batch, f"batch_slots={slots} of {a.batch}")
        check("no slot-count warning", not line, line or "none")
    else:
        if a.expect == "fewer":
            check("fewer slots than asked run", slots < a.batch, f"batch_slots={slots} of {a.batch}")
        if slots < a.batch:
            check("a line says how many fit and why", bool(line) and ("limit" in line or "fit" in line), line or "none")
            check("INFO batch_slots matches that line", said_n == slots, f"line {said_n}, INFO {slots}")
    print(f"engine log: {log_path}")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def stream(base, prompt, max_tokens, out):
    body = json.dumps({"model": "strata", "stream": True, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    out.update(t0=time.perf_counter(), times=[], text="", usage=None, error=None)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                raw = raw.decode("utf-8", "replace").strip()
                if not raw.startswith("data:") or raw == "data: [DONE]":
                    continue
                d = json.loads(raw[5:])
                if d.get("usage"):
                    out["usage"] = d["usage"]
                for c in d.get("choices") or []:
                    piece = (c.get("delta") or {}).get("content") or ""
                    if piece:
                        out["times"].append(time.perf_counter())
                        out["text"] += piece
    except Exception as e:      # noqa: BLE001 - reported as a failed check
        out["error"] = repr(e)


def rate(o):
    n = (o.get("usage") or {}).get("completion_tokens") or len(o["times"])
    if len(o["times"]) < 2:
        return n, 0.0
    return n, (n - 1) / (o["times"][-1] - o["times"][0])


def run_http(a):
    cfg = json.loads(Path(a.config).read_text())
    if a.port == 8080:
        raise SystemExit("not port 8080 (the user's service port)")
    cfg.update(exe=str(Path(a.exe).resolve()), args=with_context(cfg["args"], a.context), parallel=a.parallel,
               port=a.port, open_browser=False)
    cfg["log"] = os.path.join(tempfile.gettempdir(), "metal_slots_test_server_engine.log")
    tmp = Path(tempfile.mkdtemp(prefix="metal_slots_test_"))
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log_path = tmp / "server.log"
    base = f"http://127.0.0.1:{a.port}"
    with open(log_path, "w") as log:
        srv = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                str(cfg_path), "--port", str(a.port), "--host", "127.0.0.1"],
                               cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    try:
        t_end = time.time() + 600
        while time.time() < t_end:
            if srv.poll() is not None:
                raise SystemExit(f"the server ended - see {log_path}")
            try:
                if get(base + "/v1/status").get("loaded"):
                    break
            except Exception:       # noqa: BLE001 - not up yet
                pass
            time.sleep(2)
        st = get(base + "/v1/status")
        conc = st.get("concurrency") or {}
        check("/v1/status concurrency.serving == parallel", conc.get("serving") == a.parallel, conc)

        # warm-up, then one client alone
        stream(base, "Say hello.", 16, {})
        solo = {}
        stream(base, PROMPTS[0], a.max_tokens, solo)
        n1, r1 = rate(solo)
        check("one client streams", solo["error"] is None and n1 > 1, solo["error"] or f"{n1} tokens")

        outs = [{} for _ in range(a.parallel)]
        seen_busy, peak = [], {"n": 0, "view": None}
        done = threading.Event()

        def poll():
            while not done.is_set():
                try:
                    live = get(base + "/metrics", timeout=5).get("live") or {}
                    sl = live.get("slots") or []
                    busy = [x for x in sl if x.get("state") == "decoding"]
                    seen_busy.append(len(busy))
                    if len(busy) > peak["n"]:
                        peak.update(n=len(busy), view=sl)
                except Exception:   # noqa: BLE001 - a poll may time out under load
                    pass
                time.sleep(0.25)

        pt = threading.Thread(target=poll, daemon=True)
        pt.start()
        th = [threading.Thread(target=stream, args=(base, PROMPTS[i % len(PROMPTS)], a.max_tokens, outs[i]))
              for i in range(a.parallel)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        done.set()
        pt.join(timeout=10)
        errs = [o["error"] for o in outs if o["error"]]
        check(f"{a.parallel} clients stream", not errs and all(len(o["times"]) > 1 for o in outs), errs or "ok")
        if not errs and all(o["times"] for o in outs):
            first = max(o["times"][0] for o in outs)
            last = min(o["times"][-1] for o in outs)
            check("the streams overlap in time", first < last,
                  f"latest first token {first - outs[0]['t0']:.2f} s, earliest last token {last - outs[0]['t0']:.2f} s")
            # the window where all streams decode: tokens of each inside it / its length
            span = last - first
            per = [sum(1 for t in o["times"] if first <= t <= last) / span if span > 0 else 0.0 for o in outs]
            whole = [rate(o) for o in outs]
            print(f"1 client: {n1} tokens, {r1:.1f} tok/s")
            print(f"{a.parallel} clients, while all stream ({span:.2f} s): per stream "
                  + ", ".join(f"{x:.1f}" for x in per) + f" tok/s, total {sum(per):.1f} tok/s")
            print(f"{a.parallel} clients, each from its first to its last token: "
                  + ", ".join(f"{n} tok {r:.1f} tok/s" for n, r in whole))
        check(f"/metrics live.slots shows {a.parallel} slots decoding at one moment", peak["n"] >= a.parallel,
              f"most at once {peak['n']}: {peak['view']}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
        print(f"server log: {log_path}; engine log: {cfg['log']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="part", required=True)
    e = sub.add_parser("engine")
    h = sub.add_parser("http")
    for p in (e, h):
        p.add_argument("--exe", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--context", type=int, default=0, help="--max-context (default: the config's)")
    e.add_argument("--batch", type=int, required=True)
    e.add_argument("--expect", choices=["all", "fewer", "any"], default="any")
    h.add_argument("--port", type=int, required=True)
    h.add_argument("--parallel", type=int, default=2)
    h.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    a.exe = str(Path(a.exe).resolve())      # the engine runs in the config's cwd
    (run_engine if a.part == "engine" else run_http)(a)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
