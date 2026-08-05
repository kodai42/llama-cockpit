"""Tests for llama_cockpit.server helper functions."""

import os

import pytest

from llama_cockpit.server import (
    _parse_llama_build,
    _parse_port,
    _pid_running,
    _port_is_open,
    _classify,
)


class TestParseLlamaBuild:
    def test_simple_version(self):
        assert _parse_llama_build("llama.cpp build 3400") == 3400

    def test_colon_version(self):
        assert _parse_llama_build("build: 3362") == 3362

    def test_no_match_returns_none(self):
        assert _parse_llama_build("random text") is None

    def test_case_insensitive(self):
        assert _parse_llama_build("VERSION 5000") == 5000


class TestParsePort:
    def test_valid_port(self):
        assert _parse_port(8080, 8080, "port") == 8080

    def test_string_port(self):
        assert _parse_port("9000", 8080, "port") == 9000

    def test_none_uses_default(self):
        assert _parse_port(None, 3000, "port") == 3000

    def test_empty_string_uses_default(self):
        assert _parse_port("", 3000, "port") == 3000

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be between 1 and 65535"):
            _parse_port(0, 8080, "port")

    def test_too_high_raises(self):
        with pytest.raises(ValueError, match="must be between 1 and 65535"):
            _parse_port(70000, 8080, "port")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="must be an integer port"):
            _parse_port("abc", 8080, "port")


class TestClassify:
    def test_error_line(self):
        assert _classify("something failed") == "err"

    def test_warning_line(self):
        assert _classify("warning: low memory") == "warn"

    def test_ok_line(self):
        assert _classify("server is ready") == "ok"

    def test_info_bracket(self):
        assert _classify("[ggml] loaded") == "info"

    def test_info_llama(self):
        assert _classify("llama_init_from_result") == "info"

    def test_dim_default(self):
        assert _classify("some random log") == "dim"


class TestPidRunning:
    def test_own_pid(self):
        assert _pid_running(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 9999999 is extremely unlikely to exist
        assert _pid_running(9999999) is False


class TestPortIsOpen:
    def test_none_port(self):
        assert _port_is_open(None) is False

    def test_unlikely_port_closed(self):
        # Port 59999 is unlikely to be in use in a test env
        result = _port_is_open(59999, timeout=0.2)
        assert result is False


class TestPidfileHelpers:
    """Test pidfile read/write with actual temp files."""

    def test_write_and_read_pidfile(self, tmp_path):
        from llama_cockpit.server import _write_pidfile

        pidfile = str(tmp_path / "test.pid")
        _write_pidfile(pidfile, 12345)
        # _read_pidfile checks if PID is actually running, so 12345 will fail
        # but we can verify the file was written
        assert os.path.isfile(pidfile)
        assert open(pidfile).read().strip() == "12345"

    def test_remove_pidfile(self, tmp_path):
        from llama_cockpit.server import _remove_pidfile, _write_pidfile

        pidfile = str(tmp_path / "test.pid")
        _write_pidfile(pidfile, 12345)
        assert os.path.isfile(pidfile)
        _remove_pidfile(pidfile)
        assert not os.path.isfile(pidfile)

    def test_remove_nonexistent_pidfile(self, tmp_path):
        from llama_cockpit.server import _remove_pidfile

        # Should not raise
        _remove_pidfile(str(tmp_path / "nonexistent.pid"))
