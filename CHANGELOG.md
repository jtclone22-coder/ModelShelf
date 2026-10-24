# Changelog

## 0.1.1 (2026-10-24)

Fixes from a code review.

- No blob is called an orphan while any model manifest can't be read (its model's files would look unused), and a download in progress (`*-partial`) is never listed.

## 0.1.0 (2026-10-22)

- Installed models from Ollama's manifests: family, parameters, quantization, date, and freeable space that accounts for shared blobs.
- Finds every running Ollama server on the PC and shows loaded models via /api/ps (memory, GPU share, context, expiry).
- Remove (ollama rm) and unload (keep_alive 0, no prompt) with confirmation; orphaned blob finder; a command line and a window.
- Works on Windows in any language (Ollama ports found by their empty remote address), and talks to local Ollama servers directly, never through a proxy.
