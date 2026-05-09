#!/usr/bin/env python3
"""Serve the results/ folder on localhost:8765 and open the browser."""
import os, sys, webbrowser, http.server, threading
from pathlib import Path

PORT = 8765
ROOT = Path(__file__).parent / "results"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        pass  # silence access log


def main():
    server = http.server.HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}/index.html"
    print(f"Serving results/ at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
