"""
Minimal client for mcp_server.py — smoke test and reference for the host system.

    # server already running over HTTP (token from NEURO_MCP_TOKEN)
    python mcp_client.py --url http://127.0.0.1:8765/mcp --list
    # classify: the flat MedGemma-schema diagnosis (task: tumor | ms | stroke)
    python mcp_client.py --url http://127.0.0.1:8765/mcp --task tumor --image scan.png
    python mcp_client.py --url http://127.0.0.1:8765/mcp --task stroke --volume vol.npy
    # study-level detail: classify_slices / classify_volume (task: binary_tumor | multiclass_tumor | ms | stroke)
    python mcp_client.py --url http://gpu-host:8765/mcp --task binary_tumor --images a.png b.png
    python mcp_client.py --url http://gpu-host:8765/mcp --task stroke --volume vol.npy --modality CT --detailed

    # spawn the server as a subprocess over stdio (extra server flags after --)
    python mcp_client.py --stdio --task tumor --image scan.png -- --load_4bit

Prints progress notifications to stderr and the tool result as JSON to stdout.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

TOKEN_ENV = "NEURO_MCP_TOKEN"


def _b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _server(args, server_flags: list[str]):
    if args.stdio:
        script = str(Path(__file__).with_name("mcp_server.py"))
        return StdioServerParameters(
            command=sys.executable,
            args=[script, "--transport", "stdio", *server_flags],
            env=dict(os.environ),
            cwd=str(Path(__file__).parent),
        )
    token = os.environ.get(TOKEN_ENV)
    if not token:
        return args.url
    # Client(url) cannot set headers; pass a transport with the bearer token instead.
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    http = create_mcp_http_client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=args.timeout),
    )
    return streamable_http_client(args.url, http_client=http)


def _arguments(args) -> tuple[str, dict]:
    if args.image:
        return "classify", {"task": args.task, "image": _b64(args.image)}
    if args.volume and not args.detailed:
        return "classify", {"task": args.task, "volume": _b64(args.volume)}
    common = {
        "task": args.task,
        "pipeline": args.pipeline,
        "top_k": args.top_k,
        "aggregation": args.aggregation,
        "forest_n_agents": args.forest_n_agents,
        "debate_rounds": args.debate_rounds,
        "include_fhir": not args.no_fhir,
    }
    if args.images:
        return "classify_slices", {"images": [_b64(p) for p in args.images], **common}
    call = {"modality": args.modality, "n_slices": args.n_slices, **common}
    if args.volume_path:
        call["volume_path"] = args.volume_path
    else:
        call["volume_b64"] = _b64(args.volume)
    return "classify_volume", call


async def _progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"[progress] {progress:g}/{total if total is not None else '?'} {message or ''}", file=sys.stderr, flush=True)


async def main_async(args, server_flags) -> int:
    async with Client(_server(args, server_flags), read_timeout_seconds=args.timeout) as client:
        if args.list or not args.task:
            result = await client.call_tool("list_capabilities", {})
        else:
            name, arguments = _arguments(args)
            result = await client.call_tool(name, arguments, progress_callback=_progress)

    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        print(f"[error] {text}", file=sys.stderr)
        return 1
    payload = result.structured_content
    if payload is None:  # fall back to the text block
        payload = json.loads(result.content[0].text)
    print(json.dumps(payload, indent=2))
    return 0


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    server_flags = []
    if "--" in argv:
        split = argv.index("--")
        argv, server_flags = argv[:split], argv[split + 1:]

    p = argparse.ArgumentParser(description="Client for the neuroimaging MCP server")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    where.add_argument("--stdio", action="store_true", help="Spawn mcp_server.py over stdio")
    p.add_argument("--list", action="store_true", help="Call list_capabilities")
    p.add_argument(
        "--task",
        choices=["tumor", "binary_tumor", "multiclass_tumor", "ms", "stroke"],
        help="classify takes tumor|ms|stroke; --images/--detailed take binary_tumor|multiclass_tumor|ms|stroke",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--image", help="One PNG/JPEG → classify (flat diagnosis)")
    src.add_argument("--images", nargs="+", help="PNG/JPEG slices → classify_slices")
    src.add_argument("--volume", help="Local .npy/.npz (sent base64) → classify, or classify_volume with --detailed")
    src.add_argument("--volume_path", help="Server-side .npy/.npz (server needs --volume_dir)")
    p.add_argument("--detailed", action="store_true", help="--volume → classify_volume (study-level detail)")
    p.add_argument("--modality", choices=["CT", "MR"], default="MR")
    p.add_argument("--pipeline", choices=["standard", "forest", "debate"], default="standard")
    p.add_argument("--n_slices", type=int, default=32)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--aggregation", choices=["max_abnormal", "majority"], default="max_abnormal")
    p.add_argument("--forest_n_agents", type=int, default=3)
    p.add_argument("--debate_rounds", type=int, default=1)
    p.add_argument("--no_fhir", action="store_true", help="Omit FHIR bundles from the result")
    p.add_argument("--timeout", type=float, default=1800.0, help="Read timeout in seconds")
    args = p.parse_args(argv)
    if args.task and not (args.image or args.images or args.volume or args.volume_path) and not args.list:
        p.error("--task needs --image, --images, --volume or --volume_path")
    return args, server_flags


def _root_cause(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


def main(argv=None):
    args, server_flags = parse_args(argv)
    try:
        code = asyncio.run(main_async(args, server_flags))
    except Exception as exc:  # transport/auth failures arrive wrapped in task-group exceptions
        cause = _root_cause(exc)
        hint = "" if args.stdio else f" (is the server up at {args.url}, and is {TOKEN_ENV} right?)"
        print(f"[error] {type(cause).__name__}: {cause}{hint}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
