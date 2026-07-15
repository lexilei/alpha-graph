"""CLI for the monitor dashboard.

    python -m alpha_graph.monitor                  # build + open in a browser
    python -m alpha_graph.monitor --no-open        # build only
    python -m alpha_graph.monitor --serve          # localhost:8765, live rebuild
    python -m alpha_graph.monitor --serve --port 9000
    python -m alpha_graph.monitor --deep           # + distinct ticker counts

`--serve` re-reads the artifacts on every request, so a browser refresh during
a long run shows the current ledger and cache state; leave it open while a
build runs. It binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from loguru import logger

from alpha_graph.monitor.dashboard import OUT_PATH, build, gather, render


def _serve(port: int, deep: bool, open_browser: bool) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            try:
                body = render(gather(deep=deep)).encode("utf-8")
            except Exception as exc:
                logger.exception("render failed")
                self.send_error(500, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass  # loguru below carries what matters

    url = f"http://127.0.0.1:{port}/"
    server = HTTPServer(("127.0.0.1", port), Handler)
    logger.success(f"monitor on {url} — rebuilt on each refresh; ctrl-C to stop")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped")
    finally:
        server.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="alpha-graph research monitor")
    p.add_argument("--serve", action="store_true",
                   help="serve on localhost and rebuild on every request")
    p.add_argument("--port", type=int, default=8765, help="port for --serve")
    p.add_argument("--out", default=str(OUT_PATH), help="output path (build mode)")
    p.add_argument("--deep", action="store_true",
                   help="also read ticker columns for distinct counts (slower)")
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                   help="open a browser (default on)")
    args = p.parse_args()

    if args.serve:
        _serve(args.port, args.deep, args.open)
        return

    from pathlib import Path
    out = build(Path(args.out), deep=args.deep)
    logger.success(f"dashboard -> {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
