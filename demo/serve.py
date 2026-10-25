#!/usr/bin/env python3
"""Static dev server for the TabInfer demo.

Serves the repo root with the two cross-origin-isolation headers (COOP/COEP). WebGPU
itself does not need them, but the CPU ternary engine (cpu/) shares weights between
workers through a SharedArrayBuffer, which browsers only expose on isolated pages.
Also sends no-store so edited engine files are never served stale.

    python demo/serve.py [port]      # default 8765, then open http://127.0.0.1:8765/demo/
"""
import os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter: one line per request, no timestamps
        sys.stderr.write('%s %s\n' % (self.command, self.path))


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('PORT', 8765))
    print('TabInfer demo: http://127.0.0.1:%d/demo/  (serving %s)' % (port, ROOT), flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
