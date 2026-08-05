"""Tests for llama_cockpit.config."""


def test_llama_server_default_is_generic(monkeypatch):
    """LLAMA_SERVER must not contain a personal username path."""
    # Reload config fresh
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    import importlib
    import llama_cockpit.config as cfg

    importlib.reload(cfg)
    default = cfg.LLAMA_SERVER
    # Must not contain common username patterns
    for forbidden in ("yourname", "daniel", "user", "admin", "root"):
        assert (
            forbidden not in default.lower()
        ), f"Default path must not contain '{forbidden}'"


def test_llama_server_env_override(monkeypatch):
    """LLAMA_SERVER can be overridden via environment variable."""
    monkeypatch.setenv("LLAMA_SERVER", "/custom/path/llama-server")
    import importlib
    import llama_cockpit.config as cfg

    importlib.reload(cfg)
    assert cfg.LLAMA_SERVER == "/custom/path/llama-server"


def test_cockpit_port_default():
    import llama_cockpit.config as cfg

    assert cfg.COCKPIT_PORT == 5000


def test_llama_default_port():
    import llama_cockpit.config as cfg

    assert cfg.LLAMA_DEFAULT_PORT == 8080


def test_model_extensions():
    import llama_cockpit.config as cfg

    assert cfg.MODEL_EXTENSIONS == {".gguf", ".bin", ".ggml"}


def test_llama_min_build():
    import llama_cockpit.config as cfg

    assert isinstance(cfg._LLAMA_MIN_BUILD, int)
    assert cfg._LLAMA_MIN_BUILD > 0


def test_browse_root_is_absolute():
    import llama_cockpit.config as cfg

    assert cfg.BROWSE_ROOT.startswith("/")


def test_cockpit_host_default():
    import llama_cockpit.config as cfg

    assert cfg.COCKPIT_HOST == "127.0.0.1"


def test_llama_pidfile_default():
    import llama_cockpit.config as cfg

    assert cfg.LLAMA_PIDFILE == "/tmp/llama-server.pid"


def test_llama_log_default():
    import llama_cockpit.config as cfg

    assert cfg.LLAMA_LOG == "/tmp/llama-server.log"
