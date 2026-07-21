"""`fleetlm` command line - what a contributor actually types.

    fleetlm up                                   # a whole local fleet, one command
    fleetlm batch prompts.jsonl -o results.jsonl # the fleet's actual verb
    fleetlm bench -n 500                         # how fast is this fleet, really
    fleetlm join https://fleet.example.com --token abc123
    fleetlm doctor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import sys
import time
from urllib.parse import urlparse, urlunparse


def to_ws_url(url: str) -> str:
    """Accept the URL people paste (https://host) and aim it at the node endpoint."""
    parsed = urlparse(url if "//" in url else f"//{url}", scheme="http")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(
        parsed.scheme, "ws"
    )
    path = parsed.path.rstrip("/")
    if not path.endswith("/nodes/ws"):
        path = f"{path}/nodes/ws"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def cmd_join(args) -> int:
    from node_agent.__main__ import DEFAULT_MODEL, NodeAgent
    from node_agent.engine import create_engine

    engine = create_engine(args.engine)
    model = args.model or os.environ.get("NODE_MODEL", DEFAULT_MODEL)
    token = args.token or os.environ.get("FLEETLM_JOIN_TOKEN", "")
    ws_url = to_ws_url(args.url)

    print(f"FleetLM - joining {ws_url}")
    print(f"  engine: {engine.name}   model: {model}")
    if not token:
        print("  no join token (fine for a local fleet)")
    print()

    agent = NodeAgent(ws_url, engine, model, args.batch_size, token)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nLeaving the fleet. In-flight work will be retried elsewhere.")
    return 0


def _lan_ip() -> str:
    """Best-effort LAN address, for the join command shown to other machines."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the route
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _port_is_free(host: str, port: int) -> bool:
    """Probe the exact address uvicorn will bind.

    Deliberately no SO_REUSEADDR: that option lets a bind succeed while another
    socket holds the port, which is precisely the case being tested for.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host or "0.0.0.0", port))
            return True
        except OSError:
            return False


async def _run_up(args) -> int:
    import httpx
    import uvicorn

    from node_agent.__main__ import DEFAULT_MODEL, NodeAgent
    from node_agent.engine import create_engine

    from orchestrator.config import settings

    # Align settings with the CLI before importing the app: the orchestrator
    # logs settings.host/port on startup, and it would otherwise announce the
    # default port while actually listening on --port.
    settings.host, settings.port = args.host, args.port
    if args.token:
        settings.join_token = args.token

    from orchestrator.main import app

    base = f"http://127.0.0.1:{args.port}"
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    while not server.started and not serve_task.done():
        await asyncio.sleep(0.05)
    if serve_task.done():  # startup failed
        await serve_task
        return 1
    print(f"  orchestrator  {base}", flush=True)

    node_task = None
    if not args.no_node:
        engine = create_engine(args.engine)
        model = args.model or os.environ.get("NODE_MODEL")
        if model is None:
            model = "llama3.2" if engine.name == "ollama" else DEFAULT_MODEL
        print(f"  node          engine={engine.name} model={model}", flush=True)
        agent = NodeAgent(
            to_ws_url(base), engine, model, args.batch_size,
            args.token or os.environ.get("FLEETLM_JOIN_TOKEN", ""),
        )
        node_task = asyncio.create_task(agent.run())

        # A node is only useful once it has loaded its model and said so.
        ready = False
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _ in range(1200):  # up to ~2 minutes for a cold model load
                if node_task.done():
                    break
                try:
                    health = (await client.get(f"{base}/health")).json()
                    if health["nodes"]["ready_nodes"] >= 1:
                        ready = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        if node_task.done():
            print("\n  The node stopped during startup:")
            try:
                await node_task
            except Exception as e:
                print(f"    {e}")
            print("\n  Run `fleetlm doctor` to see what this machine can serve.")
            server.should_exit = True
            await serve_task
            return 1
        print(f"  status        {'ready' if ready else 'still loading the model'}", flush=True)

    print(flush=True)
    print(f"  Dashboard     {base}/", flush=True)
    print(f"  API base      {base}/v1", flush=True)
    print(f"  Metrics       {base}/metrics", flush=True)
    print(flush=True)
    print("  Try it:", flush=True)
    print(f"    fleetlm batch prompts.jsonl -o results.jsonl --url {base}", flush=True)
    print(flush=True)
    print("  Another machine on this network can join with:", flush=True)
    print(f"    fleetlm join http://{_lan_ip()}:{args.port}", flush=True)
    print("\n  Ctrl-C to stop.\n", flush=True)

    try:
        await serve_task
    finally:
        if node_task is not None:
            node_task.cancel()
            try:
                await node_task
            except asyncio.CancelledError:
                pass
            # Let it unwind so it can release the lease and we avoid a
            # "Task was destroyed but it is pending" warning at loop close.
            try:
                await node_task
            except asyncio.CancelledError:
                pass
    return 0


def cmd_up(args) -> int:
    """Start an orchestrator and one local node, then get out of the way."""
    if not _port_is_free(args.host, args.port):
        print(f"Port {args.port} is already in use. Pick another with --port.")
        return 1

    print("FleetLM - starting a local fleet\n")
    try:
        return asyncio.run(_run_up(args))
    except KeyboardInterrupt:
        print("\nFleet stopped.")
        return 0


def _load_requests(path: str, args) -> tuple[list[dict], list[str]]:
    """Parse a JSONL file into batch requests. Returns (requests, problems)."""
    requests: list[dict] = []
    problems: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                problems.append(f"line {lineno}: not valid JSON ({e.msg})")
                continue
            if isinstance(obj, dict) and obj.get("messages"):
                messages = obj["messages"]
            elif isinstance(obj, dict) and obj.get("prompt"):
                messages = [{"role": "user", "content": obj["prompt"]}]
            elif isinstance(obj, str):
                messages = [{"role": "user", "content": obj}]
            else:
                problems.append(f"line {lineno}: expected a 'prompt' or 'messages' field")
                continue
            requests.append({
                "messages": messages,
                "max_tokens": obj.get("max_tokens", args.max_tokens)
                if isinstance(obj, dict) else args.max_tokens,
                "temperature": obj.get("temperature", args.temperature)
                if isinstance(obj, dict) else args.temperature,
            })
    return requests, problems


def _progress(counts: dict, total: int, started: float) -> str:
    done = counts.get("completed", 0) + counts.get("failed", 0)
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    filled = int(24 * done / total) if total else 0
    bar = "#" * filled + "." * (24 - filled)
    eta_s = f"{int(eta // 60)}m{int(eta % 60):02d}s" if rate > 0 else "--"
    return (
        f"\r  [{bar}] {done}/{total}  "
        f"ok {counts.get('completed', 0)}  failed {counts.get('failed', 0)}  "
        f"running {counts.get('in_flight', 0)}  {rate:.1f}/s  ETA {eta_s}   "
    )


def cmd_batch(args) -> int:
    """Submit a JSONL file as one batch and write the results back out."""
    import httpx

    requests, problems = _load_requests(args.input, args)
    for p in problems:
        print(f"  skipped {p}")
    if not requests:
        print("Nothing to submit.")
        return 1

    base = args.url.rstrip("/")
    payload = {"requests": requests}
    if args.model:
        payload["model"] = args.model

    print(f"FleetLM - submitting {len(requests)} requests to {base}\n")
    batch_id = None
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{base}/v1/batches", json=payload)
            if r.status_code != 201:
                print(f"  submit failed ({r.status_code}): {r.text[:200]}")
                return 1
            batch = r.json()
            batch_id = batch["id"]
            total = batch["request_counts"]["total"]
            print(f"  batch {batch_id}\n")

            started = time.monotonic()
            status = batch
            # A long batch should survive a blip: tolerate a few consecutive
            # poll failures before giving up, rather than aborting on the first.
            failures = 0
            while status["status"] == "in_progress":
                time.sleep(args.poll)
                try:
                    status = client.get(f"{base}/v1/batches/{batch_id}").json()
                    failures = 0
                except httpx.HTTPError:
                    failures += 1
                    if failures >= 5:
                        raise
                    continue
                sys.stdout.write(_progress(status["request_counts"], total, started))
                sys.stdout.flush()
            sys.stdout.write(_progress(status["request_counts"], total, started) + "\n")

            body = client.get(f"{base}/v1/batches/{batch_id}/results").text
    except KeyboardInterrupt:
        # Cancel server-side rather than leaving the fleet working on units
        # nobody is waiting for.
        if batch_id:
            try:
                with httpx.Client(timeout=10.0) as c:
                    c.post(f"{base}/v1/batches/{batch_id}/cancel")
                print(f"\n  cancelled {batch_id}")
            except Exception:
                print(f"\n  could not cancel {batch_id} - cancel it manually")
        return 130
    except httpx.HTTPError as e:
        print(f"  cannot reach the orchestrator at {base}: {e}")
        return 1

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(body)

    counts = status["request_counts"]
    usage = status.get("usage", {})
    elapsed = time.monotonic() - started
    print()
    print(f"  wrote {counts['completed'] + counts['failed']} results to {args.output}")
    print(f"  {elapsed:.1f}s wall clock  ·  {usage.get('total_tokens', 0)} tokens")
    if counts.get("failed"):
        print(f"  {counts['failed']} unit(s) failed after retries - see the error field")
        return 1
    return 0


# ── bench ───────────────────────────────────────────────────────────────
#
# The claim this project rests on is that N machines finish a batch faster
# than one. That is a measurement, and it has to be one a stranger can repeat
# on their own hardware and send back - so the workload is generated here
# rather than read from a file, and is identical on every machine that runs it.

BENCH_PROMPTS = [
    "Name three uses for a paperclip.",
    "What is the capital of Japan?",
    "Explain gravity in one sentence.",
    "List two prime numbers over 50.",
    "Why is the sky blue?",
    "Give a synonym for 'rapid'.",
    "What does an orchestrator do?",
    "Summarise photosynthesis briefly.",
    "How many continents are there?",
    "Define latency in computing.",
    "What is a work queue?",
    "Name a programming language.",
]


def bench_workload(n: int, max_tokens: int) -> list[dict]:
    """A fixed workload of `n` requests, identical everywhere it is generated.

    Prompts cycle a fixed pool with the index appended, so every unit is
    distinct - a fleet that deduplicated or cached identical prompts would
    otherwise post a speedup it did not earn. Temperature is 0 so two runs are
    comparable.
    """
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{BENCH_PROMPTS[i % len(BENCH_PROMPTS)]} (#{i})",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        for i in range(n)
    ]


def bench_record(
    *, workload: dict, runs: list[float], before: dict, after: dict
) -> dict:
    """Shape one bench result, including what it fails to establish.

    Kept pure so the arithmetic is testable without a fleet: the numbers this
    produces are the ones that would be published, and a speedup ratio computed
    wrongly is worse than no number at all.
    """
    ordered = sorted(runs)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    lo, hi = (ordered[0], ordered[-1]) if ordered else (0.0, 0.0)
    nodes = after.get("nodes", [])

    def delta(key: str) -> int:
        return max(0, after.get(key, 0) - before.get(key, 0))

    caveats = []
    if len(nodes) < 2:
        caveats.append(
            "Single node: this is a baseline, not a speedup. "
            "It becomes one only when compared against a run with more machines."
        )
    if len(runs) < 3:
        caveats.append(
            "Fewer than 3 replicates: the spread is not a reliable noise floor."
        )
    spread_pct = round(100 * (hi - lo) / median, 1) if median else 0.0
    if spread_pct > 10:
        caveats.append(
            f"Run-to-run spread is {spread_pct}%, wide enough that a "
            "difference smaller than that is noise."
        )
    caveats.append(
        "Latency percentiles cover the orchestrator's whole lifetime, not just "
        "this run - start a fresh fleet before benching for clean figures."
    )

    return {
        "workload": workload,
        "fleet": {
            "nodes": len(nodes),
            "per_node": [
                {
                    "gpu": n.get("gpu"),
                    "units_completed": n.get("units_completed"),
                    "units_per_hour": n.get("units_per_hour"),
                    "tokens_per_sec": n.get("tokens_per_sec"),
                    "join_to_ready_sec": n.get("join_to_ready_sec"),
                }
                for n in nodes
            ],
        },
        "runs_sec": [round(r, 2) for r in runs],
        "median_sec": round(median, 2),
        "fastest_sec": round(lo, 2),
        "slowest_sec": round(hi, 2),
        "spread_pct": spread_pct,
        "units_per_sec": (
            round(workload["requests"] / median, 2) if median else 0.0
        ),
        "unit_latency_sec": after.get("unit_latency_sec", {}),
        "retries": delta("unit_retries"),
        "leases_reclaimed": delta("leases_reclaimed"),
        "units_failed": delta("units_failed"),
        "does_not_establish": caveats,
    }


def _print_bench(record: dict) -> None:
    fleet = record["fleet"]
    w = record["workload"]
    print()
    print("  ── result " + "─" * 52)
    print(f"  workload      {w['requests']} requests, {w['max_tokens']} max tokens, temp 0")
    print(f"  model         {w['model'] or 'fleet default'}")
    print(f"  nodes         {fleet['nodes']}")
    for n in fleet["per_node"]:
        print(
            f"    - {n['gpu']}: {n['units_completed']} units, "
            f"{n['units_per_hour']}/hour, {n['tokens_per_sec']} tok/s"
        )
    runs = "  ".join(f"{r}s" for r in record["runs_sec"])
    print(f"  runs          {runs}")
    print(
        f"  median        {record['median_sec']}s "
        f"({record['units_per_sec']} units/s, spread {record['spread_pct']}%)"
    )
    tail = record.get("unit_latency_sec", {})
    for name in ("queue", "service"):
        d = tail.get(name) or {}
        if d.get("count"):
            print(
                f"  {name:<13} p50 {d['p50']}s  p90 {d['p90']}s  "
                f"p99 {d['p99']}s  max {d['max']}s"
            )
    print(
        f"  retries       {record['retries']}   "
        f"reclaimed {record['leases_reclaimed']}   failed {record['units_failed']}"
    )
    print()
    print("  ── what this does not establish " + "─" * 31)
    for c in record["does_not_establish"]:
        print(f"    - {c}")
    print()


def cmd_bench(args) -> int:
    """Run a fixed workload against a fleet and report what actually happened."""
    import httpx

    base = args.url.rstrip("/")
    requests = bench_workload(args.n, args.max_tokens)
    payload = {"requests": requests}
    if args.model:
        payload["model"] = args.model

    print(f"FleetLM bench - {args.n} requests x {args.replicates} replicates -> {base}")

    runs: list[float] = []
    try:
        with httpx.Client(timeout=60.0) as client:
            before = client.get(f"{base}/metrics").json()
            nodes = before.get("nodes_live", 0)
            if not nodes:
                print("  no node is connected - start one with: fleetlm up")
                return 1
            print(f"  {nodes} node(s) connected\n")

            for i in range(args.replicates):
                r = client.post(f"{base}/v1/batches", json=payload)
                if r.status_code != 201:
                    print(f"  submit failed ({r.status_code}): {r.text[:200]}")
                    return 1
                status = r.json()
                batch_id = status["id"]
                total = status["request_counts"]["total"]
                started = time.monotonic()
                # The redraw only collapses on a terminal. This output exists
                # to be pasted into an issue or a reply, so off a TTY it stays
                # one line per run instead of a wall of half-finished bars.
                live = sys.stdout.isatty()
                while status["status"] == "in_progress":
                    time.sleep(args.poll)
                    status = client.get(f"{base}/v1/batches/{batch_id}").json()
                    if live:
                        sys.stdout.write(
                            f"\r  run {i + 1}/{args.replicates}"
                            + _progress(status["request_counts"], total, started)
                        )
                        sys.stdout.flush()
                runs.append(time.monotonic() - started)
                prefix = "\r" if live else ""
                print(
                    f"{prefix}  run {i + 1}/{args.replicates}  {runs[-1]:.1f}s"
                    + (" " * 60 if live else ""),
                    flush=True,
                )

            after = client.get(f"{base}/metrics").json()
    except KeyboardInterrupt:
        print("\n  interrupted - no result written")
        return 130
    except httpx.HTTPError as e:
        print(f"  cannot reach the orchestrator at {base}: {e}")
        return 1

    record = bench_record(
        workload={
            "requests": args.n,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "model": args.model,
            "replicates": args.replicates,
        },
        runs=runs,
        before=before,
        after=after,
    )
    _print_bench(record)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"  wrote {args.output} - paste this when reporting a result\n")
    return 0


def cmd_doctor(args) -> int:
    """Report whether this machine can contribute, and with what."""
    import psutil

    print("FleetLM doctor\n")
    print(f"  platform      {platform.system()} {platform.machine()}")
    print(f"  python        {sys.version.split()[0]}")

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"  memory        {ram_gb:.1f} GB")

    from node_agent.engine import OllamaEngine

    engines = []
    ollama_models: list[str] = []
    try:
        ollama_models = OllamaEngine().available_models()
        engines.append("ollama")
    except Exception:
        pass
    try:
        import mlx_lm  # noqa: F401
        engines.append("mlx")
    except ImportError:
        pass
    try:
        import llama_cpp  # noqa: F401
        engines.append("llama_cpp")
    except ImportError:
        pass
    print(f"  engines       {', '.join(engines) if engines else 'none (mock only)'}")

    if "ollama" in engines:
        print(f"  ollama models {', '.join(ollama_models) if ollama_models else 'none pulled'}")
        parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        # Batch work is only concurrent if the daemon will run requests in
        # parallel; unset, Ollama may serialise them and a wide lease gains
        # nothing.
        print(f"  ollama parallel  {parallel or 'unset - set OLLAMA_NUM_PARALLEL=4 for batch throughput'}")
        if not ollama_models:
            print("\n  No Ollama model pulled yet:  ollama pull llama3.2")
            # Only a hard stop when Ollama is the one thing that could serve;
            # an mlx/llama.cpp install can still carry this machine.
            if engines == ["ollama"]:
                return 1

    print()
    if not engines:
        print("  No inference engine found. The easiest path:")
        print("      1. install Ollama from ollama.com")
        print("      2. ollama pull llama3.2")
        print("      3. fleetlm up")
        return 1

    # Rough guidance: a 4-bit model needs ~0.6 GB per billion parameters,
    # plus headroom for KV cache and the rest of the system.
    usable = max(0.0, ram_gb - 6)
    print(f"  Usable for a model: ~{usable:.0f} GB after system headroom")
    if usable >= 18:
        print("  Suggested: an 8B 4-bit model (or larger)")
    elif usable >= 6:
        print("  Suggested: a 3B 4-bit model")
    elif usable >= 2:
        print("  Suggested: a 1B 4-bit model")
    else:
        print("  Tight on memory - this machine may struggle to contribute")
    print("\n  Ready to join:  fleetlm join <fleet-url> --token <token>")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fleetlm",
        description="Contribute your machine to a FleetLM fleet, or check whether it can.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ENGINES = ["auto", "ollama", "mlx", "llama_cpp", "mock"]

    up = sub.add_parser("up", help="start a local fleet: orchestrator + one node")
    up.add_argument("--port", type=int, default=8080)
    up.add_argument("--host", default="0.0.0.0")
    up.add_argument("--model", default=None, help="model to serve")
    up.add_argument("--token", default=None, help="require this join token")
    up.add_argument("--engine", default=os.environ.get("NODE_ENGINE", "auto"),
                    choices=ENGINES)
    up.add_argument("--batch-size", type=int,
                    default=int(os.environ.get("NODE_BATCH_SIZE", "4")))
    up.add_argument("--no-node", action="store_true",
                    help="orchestrator only; nodes join from elsewhere")
    up.set_defaults(func=cmd_up)

    batch = sub.add_parser("batch", help="run a JSONL file of prompts through the fleet")
    batch.add_argument("input", help="JSONL file: one {\"prompt\": ...} per line")
    batch.add_argument("-o", "--output", default="results.jsonl")
    batch.add_argument("--url", default=os.environ.get("FLEETLM_URL", "http://localhost:8080"))
    batch.add_argument("--model", default=None)
    batch.add_argument("--max-tokens", type=int, default=256)
    batch.add_argument("--temperature", type=float, default=0.7)
    batch.add_argument("--poll", type=float, default=1.0, help="status poll interval (s)")
    batch.set_defaults(func=cmd_batch)

    bench = sub.add_parser(
        "bench", help="measure how fast this fleet finishes a fixed workload"
    )
    bench.add_argument("-n", type=int, default=500, help="requests per run")
    bench.add_argument("--replicates", type=int, default=3, help="runs to repeat")
    bench.add_argument("--url", default=os.environ.get("FLEETLM_URL", "http://localhost:8080"))
    bench.add_argument("--model", default=None)
    bench.add_argument("--max-tokens", type=int, default=48)
    bench.add_argument("--poll", type=float, default=1.0)
    bench.add_argument("-o", "--output", default=None, help="write the record as JSON")
    bench.set_defaults(func=cmd_bench)

    join = sub.add_parser("join", help="run this machine as a fleet node")
    join.add_argument("url", help="orchestrator URL, e.g. https://fleet.example.com")
    join.add_argument("--token", default=None, help="fleet join token")
    join.add_argument("--model", default=None, help="model to serve")
    join.add_argument(
        "--engine", default=os.environ.get("NODE_ENGINE", "auto"), choices=ENGINES,
    )
    join.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("NODE_BATCH_SIZE", "4")),
        help="work units to lease at a time (default 4)",
    )
    join.set_defaults(func=cmd_join)

    doctor = sub.add_parser("doctor", help="check this machine's capability")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
