# Llama Cockpit

A lightweight, zero-overhead browser dashboard and management interface for launching, configuring, and monitoring [`llama.cpp`](https://github.com/ggml-org/llama.cpp) servers.

---

## Why Llama Cockpit?

Most local LLM GUI applications bundle monolithic execution engines and pre-packaged backends. While convenient for standard setups, these heavy wrappers frequently fail on cutting-edge hardware, non-standard toolchains, or custom builds — most notably **AMD Strix Halo (`gfx1151`)** 

**Llama Cockpit takes a backend-agnostic, native approach:**

- ⚡ **Universal Backend Support:** Works seamlessly across ROCm (including AMD Strix Halo / Ryzen AI), CUDA, Metal, and Vulkan by leveraging your native `llama.cpp` build directly.
- 🪶 **Ultra-Lightweight:** Written in standard Flask with zero heavy frontend framework overhead. Fast to load, minimal resource footprint.
- 🔍 **Dynamic Capability Detection:** Automatically probes your specific `llama-server` binary to expose only the flags your build actually supports.
- 🎛️ **Granular Command Control:** Preview the exact CLI flags being passed before launching, eliminating guesswork.


---

## Features

- **Visual Parameter Control** — Fine-tune context size, GPU offload layers, CPU threads, batch sizes, speculative decoding, sampling, and FlashAttention.
- **Auto / Custom Modes** — Rely on `llama-server` defaults or define explicit per-parameter overrides.
- **Model Browser** — Easily browse system storage and attach GGUF models with one click.
- **Per-Model Profiles** — Persist and automatically reload parameter configurations for each model.
- **Live Streaming Logs** — Real-time output stream directly from the active `llama-server` process.
- **CLI Command Preview** — Inspect the exact generated terminal command before hitting launch.
- **Capability Auto-Detection** — Automatically probes your `llama-server` binary build to enable only supported feature flags.
- **Runtime Session Recovery** — Re-attaches to an active, running `llama-server` instance upon web page refresh or browser restart.
- **Responsive UI** — Modern interface optimized for both wide desktop displays and mobile/narrow viewports.

---

## Prerequisites

- **Python 3.10+**
- **llama.cpp** — Compiled with the `llama-server` binary available on your PATH or build directory.
- **[uv](https://docs.astral.sh/uv/)** — Recommended for fast, reproducible Python environment management.

---

## Installation

```bash
git clone https://github.com/your-org/llama-cockpit.git
cd llama-cockpit
uv sync
```

## Configuration

Copy `.env.example` and edit the path to your llama-server binary:

```bash
cp .env.example .env
# Edit .env — at minimum set LLAMA_SERVER
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER` | `/path/to/llama.cpp/build/bin/llama-server` | Path to the llama-server binary |
| `LLAMA_PIDFILE` | `/tmp/llama-server.pid` | PID file for the managed llama-server process |
| `COCKPIT_HOST` | `127.0.0.1` | Host for the dashboard web server |
| `COCKPIT_PORT` | `5000` | Port for the dashboard web server |
| `LLAMA_PORT` | `8080` | Default port for llama-server |
| `COCKPIT_INDEX_FILE` | `index.html` | Path to a custom index.html (optional) |

## Usage

```bash
uv run python run.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Workflow

1. Click **Browse for model** to find and add a `.gguf` file.
2. Select the model from the list on the left.
3. Adjust parameters in the center panel (Auto or Custom per parameter).
4. Click **Save Parameters** to persist the profile.
5. Click **Launch** to start llama-server with your configuration.
6. The llama-server URL appears in the Connections panel once it's ready.

### Stopping

Click **Stop** in the dashboard, or press `Ctrl+C` in the terminal. The backend will gracefully shut down llama-server.

## Project Structure

```
llama-cockpit/
├── llama_cockpit/
│   ├── __init__.py
│   ├── config.py          # Environment-variable configuration
│   ├── server.py          # Flask backend, routes, helpers
│   └── static/
│       └── index.html     # Frontend (CSS + JS)
├── tests/
│   ├── test_config.py
│   └── test_helpers.py
├── run.py                 # Thin entry point
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Code Quality

```bash
# Format check
uv run black --check .

# Lint
uv run flake8 .

# Tests
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
