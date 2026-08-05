# Llama Cockpit

Browser dashboard for launching and configuring `llama.cpp` servers.

## Features

- **Visual parameter control** — context size, GPU offload, threads, batching, speculative decoding, sampling, and more
- **Auto/Custom modes** — let llama.cpp use defaults or set explicit values per parameter
- **Model browser** — browse your filesystem and add GGUF models with one click
- **Per-model profiles** — save and restore parameter configurations per model
- **Live log viewer** — stream llama-server output in real time
- **Command preview** — see the exact command before launching
- **Capability detection** — automatically probes your llama-server build for supported flags
- **Runtime recovery** — reconnects to a running llama-server after page reload
- **Responsive** — works on desktop and narrow screens

## Prerequisites

- **Python 3.10+**
- **llama.cpp** — compiled with `llama-server` binary available
- **[uv](https://docs.astral.sh/uv/)** — for dependency management

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
| `LLAMA_LOG` | `/tmp/llama-server.log` | Log file path |
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
