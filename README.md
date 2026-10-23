# ModelShelf

**See and tidy the local AI models on your PC.** For people who run [Ollama](https://ollama.com), ModelShelf shows:
- **Every installed model** with its family, size, parameter count and quantization.
- **What removing it would really free.** Models share files, so the honest number is often smaller than the size.
- **What's loaded right now on every Ollama server** on the machine: how much memory each uses, how much of it sits on the GPU, its context length and how long it stays loaded.

## Download

Get **ModelShelf-0.1.0-windows.zip** from [Releases](../../releases/latest), unzip it anywhere and double-click **ModelShelf.exe** in the ModelShelf folder. Nothing to install, and no Python needed.

Windows may say "Windows protected your PC" because the app is new: click **More info**, then **Run anyway**.

Prefer Python? With Python 3.9 or newer, download this repository and run `py -m modelshelf` in its folder.

Remove a model, or unload one from memory, with one confirmed click. Removing goes through Ollama's own `ollama rm`, so its index stays correct. Unloading sends no prompt, so nothing is generated.

```
py -m modelshelf list
py -m modelshelf loaded
py -m modelshelf remove llama3.3:70b-instruct-q4_K_M
py -m modelshelf unload qwen3:14b --port 11434
py -m modelshelf orphans         # blob files left by interrupted downloads
```

Example (the developer's own PC, three Ollama servers):
```
127.0.0.1:11434  nothing loaded
127.0.0.1:11435  qwen3.5:35b-a3b    21.8 GB  100% on GPU  context 32,768
127.0.0.1:11436  nothing loaded
```

Reads `OLLAMA_MODELS` if you moved the store. Nothing to install beyond Python. It is not affiliated with Ollama.

MIT License.
