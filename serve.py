"""Serve the ensemble JSON outputs on localhost:9099 with CORS so the Label Studio tab can fetch them."""
import functools
import http.server
import sys


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "http://localhost:9091")
        super().end_headers()


handler = functools.partial(H, directory=sys.argv[1])
http.server.ThreadingHTTPServer(("127.0.0.1", 9099), handler).serve_forever()
