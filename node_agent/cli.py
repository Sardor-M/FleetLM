"""`fleetlm` command line - what a contributor actually types.

    fleetlm up                                   # a whole local fleet, one command
    fleetlm batch prompts.jsonl -o results.jsonl # the fleet's actual verb
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
            while status["status"] == "in_progress":
                time.sleep(args.poll)
                status = client.get(f"{base}/v1/batches/{batch_id}").json()
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
            print("\n  No model pulled yet:  ollama pull llama3.2")
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
