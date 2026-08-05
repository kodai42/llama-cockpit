#!/usr/bin/env python3
"""
app.py — Backend for the Llama Cockpit browser dashboard.
"""

import atexit
import json
import os
import re
import shlex
import signal
import sys
import struct
import subprocess
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import requests
from flask import Flask, jsonify, request, send_from_directory

# =============================================================================
# CONFIGURATION  (all overridable via environment variables)
# =============================================================================

LLAMA_SERVER  = os.environ.get("LLAMA_SERVER",  "/home/yourname/llama.cpp/build/bin/llama-server")
LLAMA_PIDFILE = os.environ.get("LLAMA_PIDFILE", "/tmp/llama-server.pid")
LLAMA_LOG     = os.environ.get("LLAMA_LOG",     "/tmp/llama-server.log")


COCKPIT_HOST = os.environ.get("COCKPIT_HOST", "127.0.0.1")
COCKPIT_PORT = int(os.environ.get("COCKPIT_PORT", "5000"))
LLAMA_DEFAULT_PORT = int(os.environ.get("LLAMA_PORT", "8080"))

BROWSE_ROOT      = os.path.realpath(os.path.expanduser("~"))
MODEL_EXTENSIONS = {".gguf", ".bin", ".ggml"}

_MODELS_DIR  = Path.home() / ".config" / "llama-cockpit"
_MODELS_FILE = _MODELS_DIR / "models.json"
_PARAMS_FILE = _MODELS_DIR / "params.json"
_CAPS_FILE  = _MODELS_DIR / "server_caps.json"
_RUNTIME_FILE = _MODELS_DIR / "runtime.json"
_INDEX_FILE   = os.environ.get("COCKPIT_INDEX_FILE", "index.html")

_GGUF_MAX_KEY_LEN   = 4096
_GGUF_MAX_STR_LEN   = 65536
_GGUF_MAX_ARRAY_LEN = 1_000_000

# =============================================================================
# LLAMA-SERVER CAPABILITY CHECK
# =============================================================================

_LLAMA_MIN_BUILD = 3362

def _parse_llama_build(version_output: str) -> int | None:
    m = re.search(r'(?:version|build)[:\s]+(\d+)', version_output, re.IGNORECASE)
    return int(m.group(1)) if m else None

_server_caps_lock = threading.Lock()


def _server_caps_cache_key() -> dict | None:
    """Return a cache key that changes whenever the llama-server binary changes."""
    try:
        real_path = os.path.realpath(LLAMA_SERVER)
        st = os.stat(real_path)
        return {
            "path": real_path,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except Exception:
        return None


def _caps_pending(message: str = "Checking llama-server capabilities…") -> dict:
    return {
        "ok": False,
        "pending": True,
        "cached": False,
        "build": None,
        "min_build": _LLAMA_MIN_BUILD,
        "error": message,
        "supports": {},
        "help_error": None,
    }


def _decorate_caps(caps: dict, *, pending: bool = False, cached: bool = False) -> dict:
    out = dict(caps or {})
    out.setdefault("ok", False)
    out.setdefault("build", None)
    out.setdefault("min_build", _LLAMA_MIN_BUILD)
    out.setdefault("error", None)
    out.setdefault("supports", {})
    out.setdefault("help_error", None)
    out["pending"] = bool(pending)
    out["cached"] = bool(cached)
    key = _server_caps_cache_key()
    if key is not None:
        out["cache_key"] = key
    return out


def _load_cached_server_caps() -> dict | None:
    """Load cached caps only when the llama-server binary path/size/mtime matches."""
    key = _server_caps_cache_key()
    if key is None:
        return None
    try:
        data = json.loads(_CAPS_FILE.read_text())
        if data.get("cache_key") != key:
            return None
        caps = data.get("caps")
        if not isinstance(caps, dict) or caps.get("ok") is not True:
            return None
        return _decorate_caps(caps, pending=False, cached=True)
    except Exception:
        return None


def _save_server_caps_cache(caps: dict) -> None:
    """Persist only successful capability checks, atomically."""
    if not caps.get("ok"):
        return
    key = _server_caps_cache_key()
    if key is None:
        return
    try:
        _CAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cache_key": key, "caps": _decorate_caps(caps, pending=False, cached=False)}
        tmp = _CAPS_FILE.with_suffix(_CAPS_FILE.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, _CAPS_FILE)
    except Exception as exc:
        print(f"[warn] Could not save server caps cache: {exc}")


def _get_server_caps() -> dict:
    """Return a shallow copy so callers never observe a partially updated dict."""
    with _server_caps_lock:
        return dict(_server_caps)


def _set_server_caps(caps: dict) -> None:
    global _server_caps
    with _server_caps_lock:
        _server_caps = dict(caps)


def _initial_server_caps() -> dict:
    cached = _load_cached_server_caps()
    if cached is not None:
        return cached
    return _caps_pending()


def _refresh_server_caps_background() -> None:
    """Refresh capability detection after Flask startup has been unblocked."""
    global _server_caps
    fresh = _decorate_caps(_check_server_caps(), pending=False, cached=False)

    with _server_caps_lock:
        current = dict(_server_caps)
        if fresh.get("ok"):
            _server_caps = fresh
        elif current.get("ok") and current.get("cached"):
            # A valid cache for the same binary is safer than disabling launch
            # because a background refresh hit a transient probe failure.
            preserved = dict(current)
            preserved["refresh_error"] = fresh.get("error")
            _server_caps = preserved
        else:
            _server_caps = fresh

    if fresh.get("ok"):
        _save_server_caps_cache(fresh)


def _start_server_caps_probe() -> None:
    t = threading.Thread(target=_refresh_server_caps_background, name="server-caps-probe", daemon=True)
    t.start()

def _check_server_caps() -> dict:
    supports = {}
    if not os.path.isfile(LLAMA_SERVER):
        return {
            "ok": False,
            "build": None,
            "min_build": _LLAMA_MIN_BUILD,
            "error": f"llama-server binary not found at {LLAMA_SERVER}",
            "supports": supports,
        }

    def run_probe(flag: str):
        return subprocess.run(
            [LLAMA_SERVER, flag],
            capture_output=True, text=True, timeout=10,
        )

    # Keep capability detection synchronous for correctness, but run the two
    # independent llama-server probes concurrently to reduce startup latency.
    with ThreadPoolExecutor(max_workers=2) as executor:
        version_future = executor.submit(run_probe, "--version")
        help_future = executor.submit(run_probe, "--help")

        try:
            res = version_future.result()
            raw = (res.stdout + res.stderr).strip()
        except Exception as exc:
            return {
                "ok": False,
                "build": None,
                "min_build": _LLAMA_MIN_BUILD,
                "error": f"Could not run llama-server --version: {exc}",
                "supports": supports,
            }

        help_raw = ""
        help_error = None
        try:
            help_res = help_future.result()
            help_raw = help_res.stdout + help_res.stderr
        except Exception as exc:
            # Fail closed for optional flag support: empty help text means no
            # optional capability is reported as supported.
            help_error = str(exc)

    def has_any(*needles: str) -> bool:
        return any(n in help_raw for n in needles)

    supports = {
        "threadsBatch": has_any("--threads-batch", "-tb"),
        "jinja": has_any("--jinja"),
        "noJinja": has_any("--no-jinja"),
        "ui": has_any("--ui", "--no-ui"),
        "uiMcpProxy": has_any("--ui-mcp-proxy", "--webui-mcp-proxy"),
        "uiConfigFile": has_any("--ui-config-file", "--webui-config-file"),
        "chatTemplateKwargs": has_any("--chat-template-kwargs"),
        "specType": has_any("--spec-type"),
        "specDraftMin": has_any("--spec-draft-n-min"),
        "specDraftMax": has_any("--spec-draft-n-max"),
        "specDraftPMin": has_any("--spec-draft-p-min"),
        "specDraftPSplit": has_any("--spec-draft-p-split"),
        "specDraftCache": has_any("--cache-type-k-draft", "--cache-type-v-draft"),
        "draftGpuLayers": has_any("--n-gpu-layers-draft", "--gpu-layers-draft"),
        "kvCacheTypes": has_any("--cache-type-k", "--cache-type-v"),
        "kvUnified": has_any("--kv-unified", "--no-kv-unified", "-kvu", "-no-kvu"),
        "kvOffload": has_any("--kv-offload", "--no-kv-offload"),
        "cachePrompt": has_any("--cache-prompt", "--no-cache-prompt"),
        "cacheReuse": has_any("--cache-reuse"),
        "mmap": has_any("--mmap", "--no-mmap"),
        "mlock": has_any("--mlock"),
        "contBatching": has_any("--cont-batching", "--no-cont-batching"),
        "directIo": has_any("--direct-io", "--no-direct-io"),
        "numa": has_any("--numa"),
        "device": has_any("--device"),
        "perf": has_any("--perf", "--no-perf"),
    }

    build = _parse_llama_build(raw)
    if build is None:
        return {
            "ok": False,
            "build": None,
            "min_build": _LLAMA_MIN_BUILD,
            "error": f"Could not parse build number from llama-server --version output. Raw output: {raw!r}",
            "supports": supports,
            "help_error": help_error,
        }
    if build < _LLAMA_MIN_BUILD:
        return {
            "ok": False,
            "build": build,
            "min_build": _LLAMA_MIN_BUILD,
            "error": f"llama-server build b{build} is too old. --no-ui requires build b{_LLAMA_MIN_BUILD} or newer.",
            "supports": supports,
            "help_error": help_error,
        }
    return {
        "ok": True,
        "build": build,
        "min_build": _LLAMA_MIN_BUILD,
        "error": None,
        "supports": supports,
        "help_error": help_error,
    }

# =============================================================================
# RUNTIME STATE & CLEANUP
# =============================================================================

app = Flask(__name__, static_folder=".")

LOG_LLAMA: deque = deque(maxlen=500)

_llama_port: int | None = None
_llama_proc = None
_models_lock = threading.Lock()
_params_lock = threading.Lock()
_runtime_lock = threading.Lock()
_launch_lock = threading.Lock()
_stop_timer = None

def _cleanup_processes(*args):
    global _stop_timer
    if _stop_timer is not None:
        _stop_timer.cancel()
    print("\n[cleanup] Terminating llama-server...")
    _do_stop_llama()
    if args:
        sys.exit(0)
    
atexit.register(_cleanup_processes)
try:
    signal.signal(signal.SIGTERM, _cleanup_processes)
    signal.signal(signal.SIGINT, _cleanup_processes)
except Exception:
    pass

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _save_json(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        print(f"[warn] Could not save {path.name}: {exc}")

def _load_models() -> list:  return _load_json(_MODELS_FILE, [])
def _save_models(m: list):   _save_json(_MODELS_FILE, m)
def _load_params() -> dict:  return _load_json(_PARAMS_FILE, {})
def _save_params(p: dict):   _save_json(_PARAMS_FILE, p)

_models: list = _load_models()
_server_caps: dict = _initial_server_caps()
_start_server_caps_probe()

# =============================================================================
# HELPERS
# =============================================================================

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
        
def _read_pidfile(path: str) -> int | None:
    try:
        with open(path) as f:
            pid = int(f.read().strip())
        if _pid_running(pid):
            return pid
        os.remove(path)
    except Exception:
        pass
    return None

def _write_pidfile(path: str, pid: int) -> None:
    with open(path, "w") as f:
        f.write(str(pid))

def _remove_pidfile(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def _stream_log(proc, buf: deque, label: str) -> None:
    try:
        for line in proc.stdout:
            line = line.rstrip()
            buf.append({"t": _ts(), "text": line, "cls": _classify(line)})
    except Exception:
        pass
    buf.append({"t": _ts(), "text": f"[{label}] process ended.", "cls": "warn"})

def _classify(line: str) -> str:
    low = line.lower()
    if any(w in low for w in ("error", "failed", "abort", "fatal")): return "err"
    if any(w in low for w in ("warning", "warn")): return "warn"
    if any(w in low for w in ("listening", "ready", "ok", "started", "running", "health")): return "ok"
    if line.startswith("[") or "llama_" in low or "slot" in low: return "info"
    return "dim"

def _poll_health(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False
    
def _port_is_open(port: int | None, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """Return True if something is listening on host:port."""
    if port is None:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _local_url(port: int | None, path: str = "") -> str | None:
    if port is None:
        return None
    suffix = path if path.startswith("/") or not path else f"/{path}"
    return f"http://127.0.0.1:{int(port)}{suffix}"


def _health_ok(port: int | None, timeout: float = 1.0) -> bool:
    """Return True if a local service answers /health with HTTP 200."""
    if port is None:
        return False
    try:
        r = requests.get(f"http://127.0.0.1:{int(port)}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _load_runtime_state() -> dict:
    data = _load_json(_RUNTIME_FILE, {})
    return data if isinstance(data, dict) else {}


def _save_runtime_state(state: dict) -> None:
    _save_json(_RUNTIME_FILE, state if isinstance(state, dict) else {})


def _persist_runtime_service(name: str, pid: int | None, port: int | None) -> None:
    if pid is None or port is None:
        return
    try:
        pid = int(pid)
        port = int(port)
    except Exception:
        return
    with _runtime_lock:
        state = _load_runtime_state()
        state[name] = {"pid": pid, "port": port, "updated_at": int(time.time())}
        _save_runtime_state(state)


def _clear_runtime_service(name: str) -> None:
    with _runtime_lock:
        state = _load_runtime_state()
        if name in state:
            del state[name]
            _save_runtime_state(state)


def _runtime_entry(name: str) -> tuple[int | None, int | None]:
    entry = _load_runtime_state().get(name) or {}
    if not isinstance(entry, dict):
        return None, None
    try:
        pid = int(entry.get("pid")) if entry.get("pid") is not None else None
        port = int(entry.get("port")) if entry.get("port") is not None else None
    except Exception:
        return None, None
    if port is not None and not (1 <= port <= 65535):
        port = None
    return pid, port


def _service_port_valid(port: int | None, *, require_health: bool = False) -> bool:
    if port is None:
        return False
    if _health_ok(port):
        return True
    return False if require_health else _port_is_open(port)



def _recover_runtime_service(
    name: str,
    pidfile: str,
    *,
    default_port: int | None = None,
    require_health: bool = False,
) -> tuple[int | None, int | None]:
    """Recover a service PID/port from pidfile + runtime.json, validating both."""
    saved_pid, saved_port = _runtime_entry(name)
    pid = _read_pidfile(pidfile)

    if pid is None and saved_pid and _pid_running(saved_pid):
        pid = saved_pid
        try:
            _write_pidfile(pidfile, pid)
        except Exception:
            pass

    if pid is None:
        _clear_runtime_service(name)
        return None, None

    candidates: list[int] = []
    for candidate in (saved_port, default_port):
        if candidate is None:
            continue
        try:
            candidate = int(candidate)
        except Exception:
            continue
        if 1 <= candidate <= 65535 and candidate not in candidates:
            candidates.append(candidate)

    for port in candidates:
        if _service_port_valid(port, require_health=require_health):
            _persist_runtime_service(name, pid, port)
            return pid, port

    # Preserve the running PID, but do not guess a port if validation failed.
    return pid, None


def _recover_llama_runtime() -> dict:
    global _llama_port

    with _runtime_lock:
        state = _load_runtime_state()
        if any(name != "llama" for name in state):
            _save_runtime_state({"llama": state["llama"]} if "llama" in state else {})

    llama_pid, llama_port = _recover_runtime_service(
        "llama",
        LLAMA_PIDFILE,
        default_port=_llama_port or LLAMA_DEFAULT_PORT,
        require_health=False,
    )
    if llama_pid and llama_port:
        _llama_port = llama_port
    elif not llama_pid:
        _llama_port = None

    return {"llama": {"pid": llama_pid, "port": _llama_port}}


def _parse_port(value, default: int, label: str) -> int:
    if value is None or value == "":
        port = default
    else:
        try:
            port = int(value)
        except Exception:
            raise ValueError(f"{label} must be an integer port.")
    if port < 1 or port > 65535:
        raise ValueError(f"{label} must be between 1 and 65535.")
    return port


def _requested_llama_port(data: dict) -> int:
    if "llamaPort" in data:
        value = data.get("llamaPort")
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError("llama-server port is required.")
    return _parse_port(data.get("llamaPort", LLAMA_DEFAULT_PORT), LLAMA_DEFAULT_PORT, "llama-server port")




def _validate_llama_port_for_launch(port: int, data: dict) -> None:
    if port == COCKPIT_PORT:
        raise ValueError(f"Cannot launch llama-server on port {port}: that port is used by Llama Cockpit.")
    if _port_is_open(port):
        raise ValueError(f"Cannot launch llama-server: port {port} is already in use.")


def _wait_until_port_closed(port: int | None, timeout: float = 8.0) -> bool:
    """Return True if port closes within timeout."""
    if port is None:
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_is_open(port):
            return True
        time.sleep(0.2)

    return not _port_is_open(port)


def _wait_proc_exit(proc, timeout: float = 5.0) -> bool:
    """Return True if proc exits within timeout."""
    if proc is None:
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.2)

    return proc.poll() is not None


def _kill_port_owner(port: int | None) -> None:
    """Best-effort kill of whichever process owns a TCP port."""
    if port is None:
        return

    try:
        subprocess.run(
            ["bash", "-lc", f"fuser -k -TERM {int(port)}/tcp >/dev/null 2>&1 || true"],
            check=False,
        )
        time.sleep(2.0)
        subprocess.run(
            ["bash", "-lc", f"fuser -k -KILL {int(port)}/tcp >/dev/null 2>&1 || "
                            f"lsof -t -i:{int(port)} | xargs -r kill -KILL >/dev/null 2>&1 || true"],
            check=False,
        )
    except Exception as exc:
        print(f"[warn] Failed to kill TCP port owner for {port}: {exc}")

def _stop_process(proc, pidfile, log_buf, binary_path, label, kill_port=None):
    """
    Stop a service robustly.

    Important: this does NOT assume that terminating the wrapper process means
    the real service stopped. It verifies the TCP port, if one is known.
    """
    pid = _read_pidfile(pidfile)
    binary_name = os.path.basename(binary_path)
    attempted = False

    # 1. Terminate tracked wrapper/subprocess first.
    if proc:
        attempted = True
        try:
            if proc.poll() is None:
                log_buf.append({
                    "t": _ts(),
                    "text": f"Stopping {label}: sending SIGTERM to wrapper PID {proc.pid}…",
                    "cls": "warn",
                })

                # First: polite termination of the wrapper.
                proc.terminate()

                # Wait briefly.
                if not _wait_proc_exit(proc, timeout=5.0):
                    # Second: terminate the whole process group.
                    # This matters because Popen(..., start_new_session=True)
                    # creates a separate process group/session.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception as exc:
                        print(f"[warn] Failed SIGTERM process group for {label}: {exc}")

                # Wait again.
                if not _wait_proc_exit(proc, timeout=5.0):
                    # Third: force-kill the process group.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception as exc:
                        print(f"[warn] Failed SIGKILL process group for {label}: {exc}")
                        try:
                            proc.kill()
                        except Exception:
                            pass

                _wait_proc_exit(proc, timeout=2.0)

        except Exception as exc:
            print(f"[warn] Failed tracked process stop for {label}: {exc}")

    # 2. If there is an orphaned PID file PID, try to stop that too.
    if pid and _pid_running(pid):
        attempted = True
        try:
            log_buf.append({
                "t": _ts(),
                "text": f"Stopping {label}: sending SIGTERM to PID {pid}…",
                "cls": "warn",
            })

            os.kill(pid, signal.SIGTERM)

            deadline = time.time() + 5.0
            while time.time() < deadline:
                if not _pid_running(pid):
                    break
                time.sleep(0.2)

            if _pid_running(pid):
                os.kill(pid, signal.SIGKILL)

        except (ProcessLookupError, PermissionError):
            pass
        except Exception as exc:
            print(f"[warn] Failed orphan PID stop for {label}: {exc}")

    # 3. If no tracked process or PID was available, fall back to exact binary name.
    # This is intentionally not the main path, because wrappers often have a
    # different name from the actual service.
    if not attempted:
        try:
            subprocess.run(
                ["pkill", "-TERM", "-x", binary_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2.0)
            subprocess.run(
                ["pkill", "-KILL", "-x", binary_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[warn] Failed exact-name pkill for {label}: {exc}")

    # 4. Critical verification/fallback:
    # If a port is known, do not declare success while it is still open.
    port_still_open = False
    if kill_port is not None:
        if not _wait_until_port_closed(kill_port, timeout=3.0):
            port_still_open = True
            log_buf.append({
                "t": _ts(),
                "text": f"{label} port {kill_port} is still open; killing port owner…",
                "cls": "warn",
            })

            _kill_port_owner(kill_port)

            port_still_open = not _wait_until_port_closed(kill_port, timeout=5.0)

    pid_still_running = bool(pid and _pid_running(pid))
    ok = not pid_still_running and not port_still_open

    if ok:
        _remove_pidfile(pidfile)
        log_buf.append({
            "t": _ts(),
            "text": f"{label} stopped.",
            "cls": "warn",
        })
    else:
        log_buf.append({
            "t": _ts(),
            "text": f"{label} did not stop cleanly. pid_running={pid_still_running}, port_open={port_still_open}",
            "cls": "err",
        })

    return {
        "ok": ok,
        "pid": pid,
        "port": kill_port,
        "pid_running": pid_still_running,
        "port_open": port_still_open,
    }


def _stop_llama_process():
    global _llama_proc, _llama_port

    result = _stop_process(
        _llama_proc,
        LLAMA_PIDFILE,
        LOG_LLAMA,
        LLAMA_SERVER,
        "llama-server",
        kill_port=_llama_port,
    )

    if result["ok"]:
        _llama_proc = None
        _llama_port = None
        _clear_runtime_service("llama")

    return result



# =============================================================================
# GGUF PARSER
# =============================================================================

def _read_val(f, vt: int):
    _SCALAR = {
        0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2),  3: ('<h', 2),
        4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4),  7: ('<B', 1),
        10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
    }
    
    if vt in _SCALAR:
        fmt, size = _SCALAR[vt]
        return struct.unpack(fmt, f.read(size))[0]

    if vt == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        if slen > _GGUF_MAX_STR_LEN: raise ValueError(f"GGUF string length {slen} exceeds safety cap")
        return f.read(slen).decode('utf-8', 'ignore')

    if vt == 9:
        atype = struct.unpack('<I', f.read(4))[0]
        alen  = struct.unpack('<Q', f.read(8))[0]
        if alen > _GGUF_MAX_ARRAY_LEN: raise ValueError(f"GGUF array length {alen} exceeds safety cap")
        
        if atype in _SCALAR:
            _, size = _SCALAR[atype]
            skip = alen * size
            if skip > 512 * 1024 * 1024: raise ValueError(f"GGUF scalar array too large to skip ({skip} bytes)")
            f.seek(skip, os.SEEK_CUR)
            return None

        for _ in range(alen):
            _read_val(f, atype)
        return None

    raise ValueError(f"Unknown GGUF value type {vt}")

# =============================================================================
# ROUTES — static
# =============================================================================

@app.route("/")
def index():
    index_file = _INDEX_FILE if os.path.isfile(_INDEX_FILE) else "index.html"
    return send_from_directory(".", index_file)

# =============================================================================
# ROUTES — system & model info
# =============================================================================

@app.route("/api/system")
def api_system():
    return jsonify({"cpu_count": os.cpu_count() or 8})

@app.route("/api/model_info")
def api_model_info():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Invalid file path"}), 400
    real_path = os.path.realpath(path)
    if not (real_path == BROWSE_ROOT or real_path.startswith(BROWSE_ROOT + os.sep)):
        return jsonify({"error": "Access outside home directory is not permitted"}), 403
    if not os.path.isfile(real_path):
        return jsonify({"error": "Invalid file path"}), 400

    res = {"architecture": "unknown", "context_length": 8192, "block_count": 99}

    try:
        with open(real_path, 'rb') as f:
            if f.read(4) != b'GGUF': return jsonify(res)
            f.read(4)
            f.read(8)
            kv_count = struct.unpack('<Q', f.read(8))[0]

            for _ in range(min(kv_count, 1500)):
                raw_klen = f.read(8)
                if len(raw_klen) < 8: break
                key_len = struct.unpack('<Q', raw_klen)[0]
                if key_len > _GGUF_MAX_KEY_LEN: break
                key = f.read(key_len).decode('utf-8', 'ignore')

                raw_vtype = f.read(4)
                if len(raw_vtype) < 4: break
                vtype = struct.unpack('<I', raw_vtype)[0]

                try:
                    val = _read_val(f, vtype)
                except (struct.error, ValueError):
                    break

                if key == "general.architecture" and isinstance(val, str): res["architecture"] = val
                elif key.endswith(".context_length") and isinstance(val, int): res["context_length"] = val
                elif key.endswith(".block_count") and isinstance(val, int): res["block_count"] = val

    except Exception as exc:
        print(f"[warn] Failed to parse GGUF metadata for {os.path.basename(real_path)}: {exc}")

    return jsonify(res)

# =============================================================================
# ROUTES — filesystem browser
# =============================================================================

@app.route("/api/browse")
def api_browse():
    raw  = request.args.get("path", BROWSE_ROOT)
    kind = request.args.get("kind", "model")
    path = os.path.realpath(raw)

    if not (path == BROWSE_ROOT or path.startswith(BROWSE_ROOT + os.sep)):
        return jsonify({"error": "Access outside home directory is not permitted"}), 403
    if not os.path.isdir(path):
        return jsonify({"error": f"Not a directory: {path}"}), 400

    dirs, files = [], []
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
    except PermissionError:
        return jsonify({"error": f"Permission denied: {path}"}), 403

    for entry in entries:
        if entry.name.startswith(".") and kind != "config":
            continue
        try:
            is_dir  = entry.is_dir(follow_symlinks=True)
            is_file = entry.is_file(follow_symlinks=True)
        except OSError:
            continue

        if is_dir:
            dirs.append({"name": entry.name, "path": entry.path})
        elif is_file:
            ext = os.path.splitext(entry.name)[1].lower()
            if kind == "model" and ext not in MODEL_EXTENSIONS:
                continue
            try:
                size_mb = round(entry.stat(follow_symlinks=True).st_size / 1_048_576, 1)
            except OSError:
                size_mb = None
            files.append({"name": entry.name, "path": entry.path, "size_mb": size_mb})

    parent = str(os.path.dirname(path)) if path != BROWSE_ROOT else None
    return jsonify({"path": path, "parent": parent, "dirs": dirs, "files": files})

# =============================================================================
# ROUTES — model registry
# =============================================================================

@app.route("/api/models")
def api_models():
    with _models_lock:
        return jsonify([
            {"name": m["name"], "path": m["path"], "exists": os.path.isfile(m["path"])}
            for m in _models
        ])

@app.route("/api/add_model", methods=["POST"])
def api_add_model():
    data = request.json or {}
    path = data.get("path", "").strip()
    
    if not path: return jsonify({"ok": False, "error": "No path provided"}), 400
    if os.path.splitext(path)[1].lower() not in MODEL_EXTENSIONS: return jsonify({"ok": False, "error": "Unsupported extension"}), 400
    
    with _models_lock:
        if any(m["path"] == path for m in _models): return jsonify({"ok": False, "error": "Model already in list"}), 409
        name = data.get("name", "").strip() or os.path.basename(path)
        _models.append({"name": name, "path": path})
        _save_models(_models)
        
    return jsonify({"ok": True, "name": name, "path": path})

@app.route("/api/remove_model", methods=["POST"])
def api_remove_model():
    path = (request.json or {}).get("path", "").strip()
    with _models_lock:
        before = len(_models)
        _models[:] = [m for m in _models if m["path"] != path]
        if len(_models) == before: return jsonify({"ok": False, "error": "Model not found"}), 404
        _save_models(_models)
    return jsonify({"ok": True})


# =============================================================================
# PARAMETER MIGRATION & LLAMA COMMAND BUILDER
# =============================================================================

_MODE_AWARE_KEYS = {
    "ctx", "ngl", "parallel", "alias", "llamaWebui", "batch", "ubatch", "threads", "threadsBatch",
    "fa", "perf", "mmap", "mlock", "contBatching", "directIo", "numa", "device",
    "kvK", "kvV", "kvUnified", "kvOffload", "cachePrompt", "cacheReuse",
    "specType", "draftNgl", "specDraftNMin", "specDraftNMax", "specDraftPMin", "specDraftPSplit",
    "specDraftCacheK", "specDraftCacheV", "temp", "topP", "topK", "minP",
    "repeatPenalty", "presencePenalty", "frequencyPenalty", "ropeBase", "ropeScale",
}
_APP_BOOL_KEYS = set()
_TRI_KEYS = {"llamaWebui", "fa", "perf", "mmap", "mlock", "contBatching", "directIo", "kvUnified", "kvOffload", "cachePrompt"}


def _mode_entry(value, default_mode="auto") -> dict:
    if isinstance(value, dict):
        return {
            "mode": str(value.get("mode", default_mode)),
            "value": value.get("value"),
        }
    if value is None:
        return {"mode": default_mode, "value": None}
    return {"mode": "custom", "value": value}


def _migrate_saved_params(saved) -> dict:
    """Convert old primitive parameter profiles to mode-aware profiles.

    Old profiles must remain behavior-preserving, so primitive llama.cpp values are
    treated as explicit Custom overrides rather than Auto.
    """
    if not isinstance(saved, dict) or not saved:
        return {}

    migrated = False
    out = {}
    for key, value in saved.items():
        if key == "_migrated" or key in {"launchOwu", "autoBrowser", "owuPort"}:
            migrated = True
            continue
        if key in _APP_BOOL_KEYS or key in {"draftModelPath"}:
            out[key] = value
            continue
        if key == "specDraftN":
            out["specDraftNMax"] = {"mode": "custom", "value": value}
            migrated = True
            continue
        if key == "specDraftP":
            out["specDraftPMin"] = {"mode": "custom", "value": value}
            migrated = True
            continue
        if key in _MODE_AWARE_KEYS:
            if isinstance(value, dict) and "mode" in value:
                out[key] = value
            elif key in _TRI_KEYS and isinstance(value, bool):
                out[key] = {"mode": "on" if value else "off"}
                migrated = True
            else:
                out[key] = {"mode": "custom", "value": value}
                migrated = True
        else:
            out[key] = value

    if migrated:
        out["_migrated"] = True
    return out


def _entry(data: dict, key: str, default=None) -> dict:
    return _mode_entry(data.get(key, default))


def _mode(data: dict, key: str, default="auto") -> str:
    return str(_entry(data, key, {"mode": default, "value": None}).get("mode", default))


def _val(data: dict, key: str, default=None):
    return _entry(data, key, {"mode": "auto", "value": default}).get("value", default)


def _is_custom(data: dict, key: str) -> bool:
    return _mode(data, key) == "custom"


def _add_flag(cmd: list[str], flag):
    if not flag:
        return
    if isinstance(flag, (list, tuple)):
        cmd.extend(str(x) for x in flag)
    else:
        cmd.append(str(flag))


def _build_llama_command(data: dict, model: str, port: int) -> list[str]:
    cmd = [
        LLAMA_SERVER,
        "-m", model,
        "--port", str(port),
        "--host", "127.0.0.1",
    ]

    def missing_value(value) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == "")

    def require_value(key):
        value = _val(data, key)
        if missing_value(value):
            raise ValueError(f"Value required for {key}. Either enter a value or re-enable Auto.")
        return value

    def add_num(key, flag, cast=str):
        if _is_custom(data, key):
            value = require_value(key)
            try:
                cmd.extend([flag, str(cast(value))])
            except Exception as exc:
                raise ValueError(f"Invalid value for {key}: {exc}")

    def add_select(key, flag):
        if _is_custom(data, key):
            value = require_value(key)
            cmd.extend([flag, str(value)])

    def add_text(key, flag):
        if _is_custom(data, key):
            value = str(require_value(key)).strip()
            cmd.extend([flag, value])

    def add_tri(key, flag_on, flag_off=None):
        mode = _mode(data, key)
        if mode == "on":
            _add_flag(cmd, flag_on)
        elif mode == "off":
            _add_flag(cmd, flag_off)

    # Model/context and performance
    add_num("ctx", "-c", int)
    add_num("ngl", "-ngl", int)
    add_num("parallel", "-np", int)
    add_text("alias", "--alias")
    add_tri("llamaWebui", "--ui", "--no-ui")
    add_num("batch", "-b", int)
    add_num("ubatch", "-ub", int)
    add_num("threads", "-t", int)
    add_num("threadsBatch", "-tb", int)
    add_tri("fa", ["-fa", "1"], ["-fa", "0"])
    add_tri("perf", "--perf", "--no-perf")

    # Memory and cache
    add_tri("mmap", "--mmap", "--no-mmap")
    add_tri("mlock", "--mlock", None)
    add_tri("contBatching", "--cont-batching", "--no-cont-batching")
    add_tri("directIo", "--direct-io", "--no-direct-io")
    add_select("numa", "--numa")
    add_text("device", "--device")
    add_select("kvK", "--cache-type-k")
    add_select("kvV", "--cache-type-v")
    add_tri("kvUnified", "--kv-unified", "--no-kv-unified")
    add_tri("kvOffload", "--kv-offload", "--no-kv-offload")
    add_tri("cachePrompt", "--cache-prompt", "--no-cache-prompt")
    add_num("cacheReuse", "--cache-reuse", int)

    # Sampling defaults and RoPE overrides
    add_num("temp", "--temp", float)
    add_num("topP", "--top-p", float)
    add_num("topK", "--top-k", int)
    add_num("minP", "--min-p", float)
    add_num("repeatPenalty", "--repeat-penalty", float)
    add_num("presencePenalty", "--presence-penalty", float)
    add_num("frequencyPenalty", "--frequency-penalty", float)
    add_num("ropeBase", "--rope-freq-base", float)
    add_num("ropeScale", "--rope-freq-scale", float)

    # Speculative decoding
    spec_entry = _entry(data, "specType", {"mode": "auto", "value": "off"})
    spec_mode = str(spec_entry.get("mode", "auto"))
    spec_type = str(spec_entry.get("value", "off") or "off")
    draft_model = str(data.get("draftModelPath", "") or "").strip()

    if spec_mode == "custom" and spec_type in {"", "auto"}:
        raise ValueError("Value required for specType. Either select a speculative decoding type or re-enable Auto.")

    if spec_mode == "custom" and spec_type not in {"off", "none"}:
        cmd.extend(["--spec-type", spec_type])

        if spec_type == "draft":
            if not draft_model or not os.path.isfile(draft_model):
                raise ValueError("Draft model required for Draft speculative decoding")
            cmd.extend(["--model-draft", draft_model])
            add_num("draftNgl", "--n-gpu-layers-draft", int)

        if spec_type in {"draft", "draft-mtp"}:
            add_num("specDraftNMin", "--spec-draft-n-min", int)
            add_num("specDraftNMax", "--spec-draft-n-max", int)
            add_num("specDraftPMin", "--spec-draft-p-min", float)
            add_num("specDraftPSplit", "--spec-draft-p-split", float)
            add_select("specDraftCacheK", "--cache-type-k-draft")
            add_select("specDraftCacheV", "--cache-type-v-draft")

    # Other parameters. The UI manages only ui-mcp-proxy, ui-config-file,
    # template engine, template kwargs, and free-form extra shell args.
    other = data.get("other", {})
    if not isinstance(other, dict):
        other = {}

    ui_mcp_proxy_entry = other.get("uiMcpProxy")
    if not isinstance(ui_mcp_proxy_entry, dict):
        ui_mcp_proxy_entry = {"mode": "auto"}

    ui_mcp_proxy_mode = str(ui_mcp_proxy_entry.get("mode", "auto"))
    if ui_mcp_proxy_mode == "on":
        cmd.append("--ui-mcp-proxy")
    elif ui_mcp_proxy_mode == "off":
        cmd.append("--no-ui-mcp-proxy")

    ui_config_entry = other.get("uiConfigFile")
    if not isinstance(ui_config_entry, dict):
        ui_config_entry = {"mode": "off", "value": ""}

    ui_config_mode = str(ui_config_entry.get("mode", "off"))
    if ui_config_mode == "on":
        ui_config_path = str(ui_config_entry.get("value", "") or "").strip()
        if not ui_config_path:
            raise ValueError("UI config file is enabled, but no config file was selected")
        ui_config_path = os.path.realpath(os.path.expanduser(ui_config_path))
        if not os.path.isfile(ui_config_path):
            raise ValueError(f"UI config file not found: {ui_config_path}")
        cmd.extend(["--ui-config-file", ui_config_path])

    template_engine = str(other.get("templateEngine", "auto"))
    if template_engine == "on":
        cmd.append("--jinja")
    elif template_engine == "off":
        cmd.append("--no-jinja")

    kwargs = other.get("templateKwargs", {})
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs) if kwargs.strip() else {}
    if kwargs:
        if not isinstance(kwargs, dict):
            raise ValueError("Template kwargs must be a JSON object")
        cmd.extend(["--chat-template-kwargs", json.dumps(kwargs, separators=(",", ":"))])

    extra_text = str(other.get("extraArgs", "") or "").strip()
    if extra_text:
        try:
            extra = shlex.split(extra_text)
        except ValueError as exc:
            raise ValueError(f"Invalid extra shell args: {exc}")
        blocked = {"-m", "--model", "--host", "--port", "--chat-template-kwargs"}
        if any(arg in blocked for arg in extra):
            raise ValueError("Extra shell args include a blocked flag")
        cmd.extend(extra)

    return cmd

# =============================================================================
# ROUTES — status & launch
# =============================================================================

@app.route("/api/status")
def api_status():
    recovered = _recover_llama_runtime()
    llama_pid = recovered["llama"]["pid"]

    cockpit_url = _local_url(COCKPIT_PORT)
    return jsonify({
        "cockpit": {
            "port": COCKPIT_PORT,
            "url": cockpit_url,
        },
        "llama": {
            "running": llama_pid is not None,
            "pid": llama_pid,
            "port": _llama_port,
            "url": _local_url(_llama_port) if llama_pid is not None and _llama_port is not None else None,
        },
    })
    
@app.route("/api/server_caps")
def api_server_caps():
    return jsonify(_get_server_caps())
    
@app.route("/api/params")
def api_get_params():
    model_path = request.args.get("path", "").strip()
    if not model_path: return jsonify({}), 400
    return jsonify(_migrate_saved_params(_load_params().get(model_path) or {}))

@app.route("/api/params", methods=["POST"])
def api_save_params():
    data = request.json or {}
    model_path = data.get("model_path", "").strip()
    params     = data.get("params", {})
    if not model_path: return jsonify({"ok": False, "error": "No model_path provided"}), 400
    with _params_lock:
        all_params = _load_params()
        all_params[model_path] = params
        _save_params(all_params)
    return jsonify({"ok": True})
    
@app.route("/api/clear_params", methods=["POST"])
def api_clear_params():
    data = request.json or {}
    model_path = data.get("model_path", "").strip()
    if not model_path: return jsonify({"ok": False, "error": "No model_path provided"}), 400
    with _params_lock:
        all_params = _load_params()
        if model_path in all_params:
            del all_params[model_path]
            _save_params(all_params)
    return jsonify({"ok": True})

@app.route("/api/preview_command", methods=["POST"])
def api_preview_command():
    data = request.json or {}
    model = data.get("model_path", "")
    if not model:
        return jsonify({"ok": False, "error": "Model file missing"}), 400
    try:
        port = _requested_llama_port(data)
        cmd = _build_llama_command(data, model, port)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "cmd": cmd, "command": " ".join(shlex.quote(str(x)) for x in cmd)})

@app.route("/api/launch", methods=["POST"])
def api_launch():
    global _llama_proc, _llama_port
    if not _launch_lock.acquire(blocking=False): return jsonify({"ok": False, "error": "Launch already in progress"}), 409
    try:
        _recover_llama_runtime()
        if _read_pidfile(LLAMA_PIDFILE): return jsonify({"ok": False, "error": "llama-server is already running"}), 400
        caps = _get_server_caps()
        if not caps.get("ok"):
            status = 503 if caps.get("pending") else 500
            return jsonify({"ok": False, "pending": bool(caps.get("pending")), "error": caps.get("error", "check failed.")}), status
        
        data  = request.json or {}
        model = data.get("model_path", "")
        if not model or not os.path.isfile(model): return jsonify({"ok": False, "error": "Model file missing"}), 400
        
        try:
            port = _requested_llama_port(data)
            _validate_llama_port_for_launch(port, data)
            cmd = _build_llama_command(data, model, port)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        LOG_LLAMA.clear()
        LOG_LLAMA.append({"t": _ts(), "text": f"Using configured port {port}.", "cls": "info"})
        LOG_LLAMA.append({"t": _ts(), "text": "Launching: " + " ".join(shlex.quote(str(x)) for x in cmd), "cls": "dim"})

        try:
            _llama_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

        _write_pidfile(LLAMA_PIDFILE, _llama_proc.pid)
        _llama_port = port
        _persist_runtime_service("llama", _llama_proc.pid, port)
        threading.Thread(target=_stream_log, args=(_llama_proc, LOG_LLAMA, "llama-server"), daemon=True).start()
        direct_url = _local_url(port)
        return jsonify({"ok": True, "pid": _llama_proc.pid, "port": port, "direct_url": direct_url})
    finally:
        _launch_lock.release()
        
@app.route("/api/wait_llama")
def api_wait_llama():
    if _llama_port is None:
        _recover_llama_runtime()
    if _llama_port is None: return jsonify({"ok": False, "error": "not started"})
    return jsonify({"ok": _poll_health(f"http://127.0.0.1:{_llama_port}/health", 60)})
    
        

    
@app.route("/api/stop_llama", methods=["POST"])
def api_stop_llama():
    result = _stop_llama_process()

    if not result["ok"]:
        return jsonify({
            "ok": False,
            "error": "llama-server did not stop cleanly",
            "details": result,
        }), 500

    pid = result.get("pid")
    return jsonify({
        "ok": True,
        "stopped": [f"llama-server (PID {pid})"] if pid else ["llama-server"],
    })


def _do_stop_llama():
    result = _stop_llama_process()
    pid = result.get("pid")
    return [f"llama-server (PID {pid})" if pid else "llama-server"] if result.get("ok") else []

def _auto_shutdown_trigger():
    print("\n[!] No heartbeat received from dashboard for 6 seconds. Cleaning up...", flush=True)

    try:
        _do_stop_llama()
    finally:
        print("[!] Dashboard is gone. Exiting backend...", flush=True)
        os._exit(0)

@app.route("/api/heartbeat", methods=["POST"])
def api_heartbeat():
    global _stop_timer
    if _stop_timer is not None: _stop_timer.cancel()
    _stop_timer = threading.Timer(6.0, _auto_shutdown_trigger)
    _stop_timer.start()
    return jsonify({"ok": True})

@app.route("/api/logs")
def api_logs():
    since = int(request.args.get("since", 0))
    lines = list(LOG_LLAMA)
    return jsonify({"lines": lines[since:], "total": len(lines)})
    
# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Llama Cockpit — browser dashboard backend")
    print("=" * 60)
    caps = _get_server_caps()
    if caps.get("pending"):
        print(f"  Server build: capability check pending in background")
    elif not caps.get("ok"):
        print(f"  Server build: CAPABILITY CHECK FAILED")
        print(f"  Reason      : {caps.get('error')}")
    elif caps.get("cached"):
        print(f"  Server build: b{caps.get('build')} (cached; refreshing in background)")
    print("=" * 60)
    _stop_timer = threading.Timer(15.0, _auto_shutdown_trigger)
    _stop_timer.start()
    app.run(host=COCKPIT_HOST, port=COCKPIT_PORT, debug=False, threaded=True)
