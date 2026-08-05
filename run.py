#!/usr/bin/env python3
"""Thin entry point for Llama Cockpit."""

from llama_cockpit.server import app
from llama_cockpit.config import COCKPIT_HOST, COCKPIT_PORT

if __name__ == "__main__":
    app.run(host=COCKPIT_HOST, port=COCKPIT_PORT, debug=False, threaded=True)
