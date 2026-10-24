"""ModelShelf tests on a fake model store and fake server answers (no real Ollama is touched)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelshelf import store  # noqa: E402


def make_store(root: Path) -> None:
    def blob(digest: str, data: bytes) -> None:
        (root / "blobs").mkdir(parents=True, exist_ok=True)
        (root / "blobs" / digest.replace(":", "-")).write_bytes(data)

    def manifest(path: str, config: str, layers: list[tuple[str, int]]) -> None:
        target = root / "manifests" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"config": {"digest": config, "size": 50},
                                      "layers": [{"digest": d, "size": s} for d, s in layers]}))

    blob("sha256:cfg1", json.dumps({"model_family": "qwen3", "model_type": "14.8B", "file_type": "Q4_K_M"}).encode())
    blob("sha256:cfg2", json.dumps({"model_family": "qwen3", "model_type": "30.5B", "file_type": "Q4_K_M"}).encode())
    blob("sha256:cfg3", b"{}")
    manifest("registry.ollama.ai/library/qwen3/14b", "sha256:cfg1", [("sha256:weights14", 9000), ("sha256:license", 100)])
    manifest("registry.ollama.ai/library/qwen3/30b", "sha256:cfg2", [("sha256:weights30", 17000), ("sha256:license", 100)])
    manifest("registry.ollama.ai/someone/tiny/latest", "sha256:cfg3", [("sha256:tiny", 500)])
    manifest("hf.co/org/repo/Q4", "sha256:cfg3", [("sha256:hf", 700)])
    for digest in ("sha256:weights14", "sha256:weights30", "sha256:license", "sha256:tiny", "sha256:hf"):
        blob(digest, b"x")
    blob("sha256:leftover", b"x" * 42)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_store(self.root)

    def test_models_names_details_and_real_freeable_space(self) -> None:
        models = {m.name: m for m in store.read_models(self.root)}
        self.assertEqual(set(models), {"qwen3:14b", "qwen3:30b", "someone/tiny:latest", "hf.co/org/repo:Q4"})
        q14 = models["qwen3:14b"]
        self.assertEqual((q14.family, q14.parameters, q14.quantization), ("qwen3", "14.8B", "Q4_K_M"))
        self.assertEqual(q14.size, 9000 + 100 + 50)
        self.assertEqual(q14.freeable, 9000 + 50)  # the license blob is shared with qwen3:30b
        self.assertEqual(q14.shared_with, ["qwen3:30b"])
        self.assertEqual(list(models)[:1], ["qwen3:30b"])  # biggest first

    def test_orphans(self) -> None:
        self.assertEqual([p.name for p, _ in store.orphan_blobs(self.root)], ["sha256-leftover"])

    def test_no_orphans_while_a_manifest_is_unreadable_or_a_download_runs(self) -> None:
        (self.root / "blobs" / "sha256-abc-partial").write_bytes(b"x" * 10)   # a pull in progress
        self.assertEqual([p.name for p, _ in store.orphan_blobs(self.root)], ["sha256-leftover"])
        broken = self.root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "14b"
        broken.write_text('{"layers": [{"dig', encoding="utf-8")                # cut off: its blobs are unknown now
        self.assertEqual(store.orphan_blobs(self.root), [])
        self.assertEqual(store.unreadable_manifests(self.root), [broken])

    def test_loaded_models_and_unload_with_fake_server(self) -> None:
        answer = {"models": [{"name": "qwen3:14b", "size": 10_000, "size_vram": 7_500, "context_length": 8192,
                              "expires_at": "2026-10-22T14:00:00"}]}
        seen = []

        def opener(request, timeout):
            seen.append(request if isinstance(request, str) else (request.full_url, request.data))
            return io.BytesIO(json.dumps(answer).encode())

        loaded = store.loaded_models(11434, opener=opener)
        self.assertEqual((loaded[0].name, loaded[0].gpu_share, loaded[0].context), ("qwen3:14b", 0.75, 8192))
        self.assertEqual(seen[0], "http://127.0.0.1:11434/api/ps")
        self.assertIsNone(store.loaded_models(9, opener=lambda *a, **k: (_ for _ in ()).throw(OSError("refused"))))
        store.unload_model(11434, "qwen3:14b", opener=opener)
        url, body = seen[-1]
        self.assertEqual(url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(json.loads(body), {"model": "qwen3:14b", "keep_alive": 0})  # no prompt: nothing generated

    def test_remove_goes_through_ollama(self) -> None:
        calls = []

        class Done:
            returncode, stdout, stderr = 0, b"deleted 'qwen3:14b'", b""

        def runner(command, **kwargs):
            calls.append(command)
            return Done()

        self.assertIn("deleted", store.remove_model("qwen3:14b", runner=runner))
        self.assertEqual(calls[0], ["ollama", "rm", "qwen3:14b"])
        with self.assertRaises(ValueError):
            store.remove_model("qwen3 & del *", runner=runner)

    def test_server_discovery(self) -> None:
        class Out:
            def __init__(self, text):
                self.stdout = text.encode()

        def runner(command, **kwargs):
            if command[0] == "tasklist":
                return Out('"ollama.exe","100","Console","1","1 K"\n"python.exe","200","Console","1","1 K"\n')
            return Out("  TCP    127.0.0.1:11434   0.0.0.0:0   LISTENING   100\n"
                       "  TCP    127.0.0.1:11435   0.0.0.0:0   LISTENING   100\n"
                       "  TCP    127.0.0.1:8000    0.0.0.0:0   LISTENING   200\n")

        self.assertEqual(store.ollama_ports(runner), [11434, 11435])


if __name__ == "__main__":
    unittest.main()
