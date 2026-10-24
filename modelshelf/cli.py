"""Command line: list, loaded, remove, unload, orphans, gui."""
from __future__ import annotations

import argparse
import sys
import time

from . import __version__, store


def cmd_list(args) -> int:
    models = store.read_models()
    total = sum(m.size for m in models)
    print(f"{len(models)} models, {store.human(total)} in {store.models_root()}\n")
    for m in models:
        shared = f"  shares files with {', '.join(m.shared_with)}" if m.shared_with else ""
        print(f"{store.human(m.size):>9}  {m.name:<42} {m.family:<11} {m.parameters:<7} {m.quantization:<7} "
              f"added {time.strftime('%Y-%m-%d', time.localtime(m.modified))}{shared}")
    return 0


def cmd_loaded(args) -> int:
    for port in store.ollama_ports():
        loaded = store.loaded_models(port)
        if loaded is None:
            print(f"127.0.0.1:{port}  (not answering)")
            continue
        if not loaded:
            print(f"127.0.0.1:{port}  nothing loaded")
        for item in loaded:
            print(f"127.0.0.1:{port}  {item.name:<40} {store.human(item.size):>9}  {item.gpu_share:>4.0%} on GPU  "
                  f"context {item.context:,}")
    return 0


def cmd_remove(args) -> int:
    model = next((m for m in store.read_models() if m.name == args.name), None)
    if model is None:
        print(f"No model called {args.name!r}.")
        return 1
    if not args.yes and input(f"Remove {model.name} (frees {store.human(model.freeable)})? [y/N] ").lower() not in ("y", "yes"):
        print("Kept.")
        return 0
    print(store.remove_model(model.name))
    return 0


def cmd_unload(args) -> int:
    if not args.yes and input(f"Unload {args.name} from the server on port {args.port}? Anything using it will "
                              "have to load it again. [y/N] ").lower() not in ("y", "yes"):
        print("Left loaded.")
        return 0
    print(store.unload_model(args.port, args.name))
    return 0


def cmd_orphans(args) -> int:
    unreadable = store.unreadable_manifests()
    if unreadable:
        print(f"{len(unreadable)} model manifest(s) can't be read (e.g. {unreadable[0]}), so no blob can be called unused "
              "right now: their models' files would look like leftovers.")
        return 1
    orphans = store.orphan_blobs()
    for path, size in orphans:
        print(f"{store.human(size):>9}  {path}")
    print(f"{len(orphans)} unreferenced blob(s), {store.human(sum(s for _, s in orphans))}" if orphans
          else "No leftover blobs.")
    return 0


def cmd_gui(args) -> int:
    from .gui import main as gui_main
    return gui_main()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="modelshelf", description="See and tidy your local Ollama models.")
    parser.add_argument("--version", action="version", version=f"ModelShelf {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="installed models").set_defaults(func=cmd_list)
    sub.add_parser("loaded", help="models in memory on each Ollama server").set_defaults(func=cmd_loaded)
    p = sub.add_parser("remove", help="remove a model (asks first)")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_remove)
    p = sub.add_parser("unload", help="free a loaded model's memory (asks first)")
    p.add_argument("name")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_unload)
    sub.add_parser("orphans", help="blob files no model uses").set_defaults(func=cmd_orphans)
    sub.add_parser("gui", help="open the window").set_defaults(func=cmd_gui)
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        return cmd_gui(args)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
