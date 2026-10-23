"""Ollama's model store on disk, and the running Ollama servers. Reading only; changes go through Ollama itself.

Models share blobs (a fine-tune and its base, or two tags of one model), so the space a removal really
frees is only the blobs no other model uses. That's what `freeable` reports.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Model:
    name: str
    size: int
    family: str = ""
    parameters: str = ""
    quantization: str = ""
    modified: float = 0.0
    blobs: tuple = ()
    freeable: int = 0
    shared_with: list[str] = field(default_factory=list)


def models_root() -> Path:
    configured = os.environ.get("OLLAMA_MODELS", "").strip()
    return Path(configured) if configured else Path(os.path.expanduser("~")) / ".ollama" / "models"


def _name_from(parts: tuple) -> str:
    """manifests/<registry>/<namespace>/<model>/<tag> -> 'model:tag' (or 'ns/model:tag', 'registry/ns/model:tag')."""
    registry, namespace, model, tag = parts[-4], parts[-3], parts[-2], parts[-1]
    if registry == "registry.ollama.ai":
        return f"{model}:{tag}" if namespace == "library" else f"{namespace}/{model}:{tag}"
    return f"{registry}/{namespace}/{model}:{tag}"


def read_models(root: Path | None = None) -> list[Model]:
    root = root or models_root()
    manifests = root / "manifests"
    models: list[Model] = []
    if not manifests.is_dir():
        return models
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(manifests).parts
        if len(parts) < 4:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = list(manifest.get("layers") or [])
        config = manifest.get("config") or {}
        blobs = {layer["digest"]: int(layer.get("size") or 0) for layer in layers + [config] if layer.get("digest")}
        model = Model(_name_from(parts), sum(blobs.values()), modified=path.stat().st_mtime, blobs=tuple(blobs))
        model._sizes = blobs  # type: ignore[attr-defined]
        try:
            info = json.loads((root / "blobs" / config["digest"].replace(":", "-")).read_text(encoding="utf-8"))
            model.family = info.get("model_family", "")
            model.parameters = info.get("model_type", "")
            model.quantization = info.get("file_type", "")
        except (KeyError, OSError, ValueError):
            pass
        models.append(model)
    users: dict[str, list[str]] = {}
    for model in models:
        for digest in model.blobs:
            users.setdefault(digest, []).append(model.name)
    for model in models:
        sizes = model._sizes  # type: ignore[attr-defined]
        model.freeable = sum(size for digest, size in sizes.items() if len(users[digest]) == 1)
        model.shared_with = sorted({other for digest in model.blobs for other in users[digest] if other != model.name})
    models.sort(key=lambda m: m.size, reverse=True)
    return models


def orphan_blobs(root: Path | None = None) -> list[tuple[Path, int]]:
    """Blob files no manifest references (left by interrupted pulls or removed models)."""
    root = root or models_root()
    referenced = {d.replace(":", "-") for m in read_models(root) for d in m.blobs}
    blobs = root / "blobs"
    if not blobs.is_dir() or not referenced:
        return []
    return [(p, p.stat().st_size) for p in blobs.iterdir()
            if p.is_file() and p.name.startswith("sha256-") and p.name not in referenced]


# ----------------------------------------------------------------------------- running servers
@dataclass
class Loaded:
    server: str
    name: str
    size: int
    size_vram: int
    context: int
    expires: str

    @property
    def gpu_share(self) -> float:
        return self.size_vram / self.size if self.size else 0.0


def ollama_ports(runner=subprocess.run) -> list[int]:
    """Ports on which an ollama process is listening (several servers can run side by side)."""
    try:
        tasks = runner(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, timeout=20, creationflags=NO_WINDOW)
        netstat = runner(["netstat", "-ano", "-p", "TCP"], capture_output=True, timeout=20, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return [11434]
    pids = {int(m.group(2)) for m in re.finditer(r'"([^"]+)","(\d+)"', tasks.stdout.decode("utf-8", "replace"))
            if m.group(1).lower() == "ollama.exe"}
    ports = set()
    for line in netstat.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        # listening rows have no remote end ("LISTENING" itself is translated on non-English Windows)
        if len(parts) >= 5 and parts[2] in ("0.0.0.0:0", "[::]:0") and parts[4].isdigit() and int(parts[4]) in pids:
            ports.add(int(parts[1].rsplit(":", 1)[1]))
    return sorted(ports) or [11434]


# Straight to 127.0.0.1: urllib would otherwise send local requests through a system or HTTP_PROXY proxy.
open_direct = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def loaded_models(port: int, opener=open_direct, timeout: float = 3.0) -> list[Loaded] | None:
    """GET /api/ps: a status read that never loads or runs a model. None if the server doesn't answer."""
    try:
        with opener(f"http://127.0.0.1:{port}/api/ps", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return [Loaded(f"127.0.0.1:{port}", m.get("name", ""), int(m.get("size") or 0), int(m.get("size_vram") or 0),
                   int(m.get("context_length") or 0), m.get("expires_at", "")) for m in data.get("models") or []]


def remove_model(name: str, runner=subprocess.run) -> str:
    """`ollama rm` keeps Ollama's own index right. Callers confirm with the user first."""
    if not re.fullmatch(r"[A-Za-z0-9._/:-]+", name or ""):
        raise ValueError("not a model name")
    result = runner(["ollama", "rm", name], capture_output=True, timeout=120, creationflags=NO_WINDOW)
    text = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
    if result.returncode != 0:
        raise RuntimeError(text or "ollama rm failed")
    return text or f"Removed {name}."


def unload_model(port: int, name: str, opener=open_direct) -> str:
    """Free a model's memory now (keep_alive 0, no prompt: nothing is generated). Callers confirm first."""
    body = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    with opener(request, timeout=60) as response:
        response.read()
    return f"Unloaded {name} from the server on port {port}."


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
