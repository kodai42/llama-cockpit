"""Configuration for Llama Cockpit.

All settings are overridable via environment variables.
"""

import os
from pathlib import Path

# ─── llama-server ──────────────────────────────────────────────────────────────

LLAMA_SERVER = os.environ.get(
    "LLAMA_SERVER",
    "/path/to/llama.cpp/build/bin/llama-server",
)
LLAMA_PIDFILE = os.environ.get("LLAMA_PIDFILE", "/tmp/llama-server.pid")
LLAMA_LOG = os.environ.get("LLAMA_LOG", "/tmp/llama-server.log")

# ─── Cockpit server ────────────────────────────────────────────────────────────

COCKPIT_HOST = os.environ.get("COCKPIT_HOST", "127.0.0.1")
COCKPIT_PORT = int(os.environ.get("COCKPIT_PORT", "5000"))
LLAMA_DEFAULT_PORT = int(os.environ.get("LLAMA_PORT", "8080"))

# ─── Filesystem & registry ─────────────────────────────────────────────────────

BROWSE_ROOT = os.path.realpath(os.path.expanduser("~"))
MODEL_EXTENSIONS = {".gguf", ".bin", ".ggml"}

_MODELS_DIR = Path.home() / ".config" / "llama-cockpit"
_MODELS_FILE = _MODELS_DIR / "models.json"
_PARAMS_FILE = _MODELS_DIR / "params.json"
_CAPS_FILE = _MODELS_DIR / "server_caps.json"
_RUNTIME_FILE = _MODELS_DIR / "runtime.json"
_INDEX_FILE = os.environ.get("COCKPIT_INDEX_FILE", "index.html")

# ─── GGUF parser safety caps ───────────────────────────────────────────────────

_GGUF_MAX_KEY_LEN = 4096
_GGUF_MAX_STR_LEN = 65536
_GGUF_MAX_ARRAY_LEN = 1_000_000

# ─── llama-server capability check ─────────────────────────────────────────────

_LLAMA_MIN_BUILD = 3362
