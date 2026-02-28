"""Command-line interface for grafeo-memory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grafeo-memory",
        description="AI memory layer powered by GrafeoDB.",
    )
    parser.add_argument("--model", "-m", help="pydantic-ai model string (env: GRAFEO_MEMORY_MODEL)")
    parser.add_argument("--db", "-d", help="Database path (env: GRAFEO_MEMORY_DB)")
    parser.add_argument("--user", "-u", help="User ID (env: GRAFEO_MEMORY_USER)")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output as JSON")
    parser.add_argument("--yolo", action="store_true", help="Enable all features (importance, vision, usage tracking)")
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a memory from text")
    p_add.add_argument("text", help="Text to add (use '-' for stdin)")
    p_add.add_argument("--type", "-t", dest="memory_type", default="semantic", choices=["semantic", "procedural"])
    p_add.add_argument("--no-infer", action="store_true", help="Store raw without LLM extraction")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=10, help="Number of results (default: 10)")
    p_search.add_argument("--type", "-t", dest="memory_type", choices=["semantic", "procedural"])

    # list
    p_list = sub.add_parser("list", help="List all memories")
    p_list.add_argument("--type", "-t", dest="memory_type", choices=["semantic", "procedural"])

    # update
    p_update = sub.add_parser("update", help="Update a memory's text")
    p_update.add_argument("memory_id", help="Memory ID to update")
    p_update.add_argument("text", help="New text")

    # delete
    p_delete = sub.add_parser("delete", help="Delete memories")
    p_delete.add_argument("memory_id", nargs="?", help="Memory ID to delete")
    p_delete.add_argument("--all", dest="delete_all", action="store_true", help="Delete all memories for user")
    p_delete.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    # history
    p_history = sub.add_parser("history", help="Show change history for a memory")
    p_history.add_argument("memory_id", help="Memory ID")

    # summarize
    p_summarize = sub.add_parser("summarize", help="Consolidate old memories")
    p_summarize.add_argument("--preserve-recent", type=int, default=5, help="Keep N most recent (default: 5)")
    p_summarize.add_argument("--batch-size", type=int, default=20, help="Memories per LLM batch (default: 20)")

    return parser


def _resolve(args_val: str | None, env_key: str, default: str | None = None) -> str | None:
    """Resolve config: CLI flag > env var > default."""
    if args_val is not None:
        return args_val
    return os.environ.get(env_key, default)


def _create_embedder(model: str):
    """Auto-detect and create the appropriate embedder based on the model string."""
    provider = model.split(":")[0] if ":" in model else "openai"

    if provider == "mistral":
        try:
            from mistralai import Mistral
        except ImportError:
            print("Error: mistralai package not installed.", file=sys.stderr)
            print("Install it with: uv add grafeo-memory[mistral]", file=sys.stderr)
            sys.exit(1)
        from .embedding import MistralEmbedder

        api_key = os.environ.get("MISTRAL_API_KEY")
        return MistralEmbedder(Mistral(api_key=api_key))

    # Default to OpenAI for openai, anthropic, groq, etc.
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed.", file=sys.stderr)
        print("Install it with: uv add grafeo-memory[openai]", file=sys.stderr)
        sys.exit(1)
    from .embedding import OpenAIEmbedder

    return OpenAIEmbedder(OpenAI())


def _create_manager(args: argparse.Namespace):
    """Build a MemoryManager from CLI args + env vars."""
    from .manager import MemoryManager
    from .types import MemoryConfig

    model = _resolve(args.model, "GRAFEO_MEMORY_MODEL", "openai:gpt-4o-mini") or "openai:gpt-4o-mini"
    db_path = _resolve(args.db, "GRAFEO_MEMORY_DB")
    user_id = _resolve(args.user, "GRAFEO_MEMORY_USER", "default") or "default"

    embedder = _create_embedder(model)
    if args.yolo:
        config = MemoryConfig.yolo(db_path=db_path, user_id=user_id)
    else:
        config = MemoryConfig(db_path=db_path, user_id=user_id)
    return MemoryManager(model, config, embedder=embedder)


# --- Output helpers ---


def _serialize(obj: Any) -> dict:
    """Convert a dataclass to a JSON-friendly dict, resolving Enums to their values."""
    from enum import Enum

    d = asdict(obj)
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in d.items()}


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def _print_events(events: list, *, json_mode: bool) -> None:
    if json_mode:
        _print_json({"events": [_serialize(e) for e in events]})
        return
    if not events:
        print("No changes.")
        return
    for e in events:
        action = e.action.value.upper()
        mid = e.memory_id or "?"
        line = f"  [{action}] {e.text}  (id: {mid})"
        if e.old_text:
            line += f"\n    was: {e.old_text}"
        print(line)


def _print_results(results: list, *, json_mode: bool) -> None:
    if json_mode:
        _print_json({"results": [_serialize(r) for r in results]})
        return
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.score:.3f}] {r.text}  (id: {r.memory_id})")


# --- Command handlers ---


def _cmd_add(manager, args: argparse.Namespace) -> None:
    text = sys.stdin.read().strip() if args.text == "-" else args.text
    if not text:
        print("Error: no text provided.", file=sys.stderr)
        sys.exit(1)

    events = manager.add(text, memory_type=args.memory_type, infer=not args.no_infer)
    _print_events(list(events), json_mode=args.output_json)


def _cmd_search(manager, args: argparse.Namespace) -> None:
    results = manager.search(args.query, k=args.k, memory_type=args.memory_type)
    _print_results(list(results), json_mode=args.output_json)


def _cmd_list(manager, args: argparse.Namespace) -> None:
    results = manager.get_all(memory_type=args.memory_type)
    if args.output_json:
        _print_json({"memories": [_serialize(r) for r in results]})
        return
    if not results:
        print("No memories.")
        return
    for r in results:
        mtype = f" [{r.memory_type}]" if r.memory_type else ""
        print(f"  {r.memory_id}. {r.text}{mtype}")


def _cmd_update(manager, args: argparse.Namespace) -> None:
    event = manager.update(args.memory_id, args.text)
    _print_events([event], json_mode=args.output_json)


def _cmd_delete(manager, args: argparse.Namespace) -> None:
    if args.delete_all:
        if not args.yes:
            user_id = _resolve(args.user, "GRAFEO_MEMORY_USER", "default")
            answer = input(f"Delete ALL memories for user '{user_id}'? [y/N] ")
            if answer.lower() not in ("y", "yes"):
                print("Aborted.")
                return
        count = manager.delete_all()
        if args.output_json:
            _print_json({"deleted_count": count})
        else:
            print(f"Deleted {count} memories.")
    elif args.memory_id:
        ok = manager.delete(args.memory_id)
        if args.output_json:
            _print_json({"deleted": ok, "memory_id": args.memory_id})
        else:
            print(f"Deleted memory {args.memory_id}." if ok else f"Memory {args.memory_id} not found.")
    else:
        print("Error: provide a memory_id or use --all.", file=sys.stderr)
        sys.exit(1)


def _cmd_history(manager, args: argparse.Namespace) -> None:
    entries = manager.history(args.memory_id)
    if args.output_json:
        _print_json({"history": [asdict(e) for e in entries]})
        return
    if not entries:
        print("No history.")
        return
    for entry in entries:
        print(f"  [{entry.event}] at {entry.timestamp}")
        if entry.old_text:
            print(f"    old: {entry.old_text}")
        if entry.new_text:
            print(f"    new: {entry.new_text}")


def _cmd_summarize(manager, args: argparse.Namespace) -> None:
    events = manager.summarize(preserve_recent=args.preserve_recent, batch_size=args.batch_size)
    _print_events(list(events), json_mode=args.output_json)
    if not args.output_json and events:
        added = sum(1 for e in events if e.action.value == "add")
        deleted = sum(1 for e in events if e.action.value == "delete")
        print(f"\nSummary: {added} consolidated, {deleted} removed.")


# --- Entry point ---

_COMMANDS = {
    "add": _cmd_add,
    "search": _cmd_search,
    "list": _cmd_list,
    "update": _cmd_update,
    "delete": _cmd_delete,
    "history": _cmd_history,
    "summarize": _cmd_summarize,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from grafeo_memory import __version__

        print(f"grafeo-memory {__version__}")
        return

    if not args.command:
        parser.print_help()
        return

    manager = _create_manager(args)
    try:
        handler = _COMMANDS[args.command]
        handler(manager, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if args.output_json:
            _print_json({"error": str(exc)})
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        manager.close()
