"""Window: installed models (size, what removal frees, remove) and what's loaded on each server (unload)."""
from __future__ import annotations

import queue
import sys
import threading
import time

from . import __version__, store


def main() -> int:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title(f"ModelShelf {__version__}: your Ollama models")
    root.geometry("1080x640")
    messages: queue.Queue = queue.Queue()
    state = {"models": {}, "loaded": {}}

    summary = ttk.Label(root, text="Reading models…", font=("Segoe UI", 11, "bold"), anchor="w")
    summary.pack(fill="x", padx=12, pady=(10, 4))
    installed = ttk.Treeview(root, columns=("size", "frees", "family", "params", "quant", "added", "shared"),
                             show="tree headings", height=12)
    installed.heading("#0", text="Model")
    installed.column("#0", width=300)
    for column, title, width in (("size", "Size", 80), ("frees", "Removing frees", 110), ("family", "Family", 90),
                                 ("params", "Params", 70), ("quant", "Quant", 70), ("added", "Added", 90),
                                 ("shared", "Shares files with", 230)):
        installed.heading(column, text=title)
        installed.column(column, width=width, anchor="w")
    installed.pack(fill="both", expand=True, padx=12)
    row = ttk.Frame(root)
    row.pack(fill="x", padx=12, pady=6)
    ttk.Label(root, text="Loaded right now", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12)
    loaded = ttk.Treeview(root, columns=("server", "size", "gpu", "context", "until"), show="tree headings", height=5)
    loaded.heading("#0", text="Model")
    loaded.column("#0", width=300)
    for column, title, width in (("server", "Server", 140), ("size", "Memory", 90), ("gpu", "On GPU", 80),
                                 ("context", "Context", 90), ("until", "Kept until", 220)):
        loaded.heading(column, text=title)
        loaded.column(column, width=width, anchor="w")
    loaded.pack(fill="x", padx=12)
    row2 = ttk.Frame(root)
    row2.pack(fill="x", padx=12, pady=(6, 10))

    def background(task, kind) -> None:
        def work():
            try:
                messages.put((kind, task()))
            except Exception as exc:
                messages.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def gather_loaded():
        return [(port, store.loaded_models(port)) for port in store.ollama_ports()]

    def refresh() -> None:
        background(store.read_models, "models")
        background(gather_loaded, "loaded")

    def remove() -> None:
        model = state["models"].get(installed.focus())
        if model and messagebox.askyesno("ModelShelf", f"Remove {model.name}?\n\nThis frees {store.human(model.freeable)}. "
                                         "You can pull it again later.", icon="warning"):
            background(lambda: store.remove_model(model.name), "done")

    def unload() -> None:
        item = state["loaded"].get(loaded.focus())
        if item and messagebox.askyesno("ModelShelf", f"Unload {item.name} from {item.server}?\n\nAnything using this "
                                        "model (a chat app, an assistant) will wait while it loads again.", icon="warning"):
            background(lambda: store.unload_model(int(item.server.rsplit(":", 1)[1]), item.name), "done")

    ttk.Button(row, text="Refresh", command=refresh).pack(side="left")
    ttk.Button(row, text="Remove model…", command=remove).pack(side="left", padx=6)
    ttk.Button(row2, text="Unload from memory…", command=unload).pack(side="left")

    def pump() -> None:
        while True:
            try:
                kind, payload = messages.get_nowait()
            except queue.Empty:
                break
            if kind == "models":
                installed.delete(*installed.get_children())
                state["models"] = {}
                for index, m in enumerate(payload):
                    iid = f"m{index}"
                    state["models"][iid] = m
                    installed.insert("", "end", iid=iid, text=m.name, values=(
                        store.human(m.size), store.human(m.freeable), m.family, m.parameters, m.quantization,
                        time.strftime("%Y-%m-%d", time.localtime(m.modified)), ", ".join(m.shared_with)))
                summary.configure(text=f"{len(payload)} models, {store.human(sum(m.size for m in payload))} in "
                                       f"{store.models_root()}")
            elif kind == "loaded":
                loaded.delete(*loaded.get_children())
                state["loaded"] = {}
                for port, items in payload:
                    if items is None:
                        loaded.insert("", "end", text="(server not answering)", values=(f"127.0.0.1:{port}", "", "", "", ""))
                        continue
                    for index, item in enumerate(items):
                        iid = f"{port}-{index}"
                        state["loaded"][iid] = item
                        loaded.insert("", "end", iid=iid, text=item.name, values=(
                            item.server, store.human(item.size), f"{item.gpu_share:.0%}", f"{item.context:,}", item.expires))
            elif kind == "done":
                messagebox.showinfo("ModelShelf", payload)
                refresh()
            elif kind == "error":
                messagebox.showerror("ModelShelf", payload)
        root.after(150, pump)

    refresh()
    root.after(150, pump)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
