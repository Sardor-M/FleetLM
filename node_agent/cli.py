"""`fleetlm` command line — what a contributor actually types.

    fleetlm join https://fleet.example.com --token abc123
    fleetlm join http://localhost:8080 --model mlx-community/Llama-3.2-3B-Instruct-4bit
    fleetlm doctor
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
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

    print(f"FleetLM — joining {ws_url}")
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


def cmd_doctor(args) -> int:
    """Report whether this machine can contribute, and with what."""
    import psutil

    print("FleetLM doctor\n")
    print(f"  platform      {platform.system()} {platform.machine()}")
    print(f"  python        {sys.version.split()[0]}")

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"  memory        {ram_gb:.1f} GB")

    engines = []
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

    print()
    if not engines:
        print("  No inference engine installed. On Apple silicon:")
        print("      pip install mlx-lm")
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
        print("  Tight on memory — this machine may struggle to contribute")
    print("\n  Ready to join:  fleetlm join <fleet-url> --token <token>")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fleetlm",
        description="Contribute your machine to a FleetLM fleet, or check whether it can.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    join = sub.add_parser("join", help="run this machine as a fleet node")
    join.add_argument("url", help="orchestrator URL, e.g. https://fleet.example.com")
    join.add_argument("--token", default=None, help="fleet join token")
    join.add_argument("--model", default=None, help="model to serve")
    join.add_argument(
        "--engine", default=os.environ.get("NODE_ENGINE", "auto"),
        choices=["auto", "mlx", "llama_cpp", "mock"],
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
