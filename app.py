"""CLI and HTTP entry point.

    python app.py "How do I reset my password?"
    echo '{"question": "How do I reset my password?"}' | python app.py
    python app.py --serve --port 8000     # POST /ask {"question": "..."}
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from supportqa import DEFAULT_KB, SupportQA
from supportqa.kb import KBError


def make_server(qa: SupportQA, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            data = json.dumps(body, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "documents": len(qa.docs)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                question = body["question"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": 'expected JSON body {"question": "string"}'})
            self._send(200, qa.ask(question))

        def log_message(self, fmt, *args):
            print(f"[http] {fmt % args}", file=sys.stderr)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description="Grounded support-answer service")
    p.add_argument("question", nargs="?", help="question to answer (or JSON on stdin)")
    p.add_argument("--kb", default=os.environ.get("KB_PATH", DEFAULT_KB), help="path to kb.json")
    p.add_argument("--mode", choices=["rule", "llm"], default=os.environ.get("ANSWER_MODE", "rule"),
                   help="answer composer (llm falls back to rule if unavailable)")
    p.add_argument("--serve", action="store_true", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    try:
        qa = SupportQA(args.kb, mode=args.mode)
    except KBError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.serve:
        server = make_server(qa, args.host, args.port)
        print(f"Serving on http://{args.host}:{server.server_port}  (POST /ask, GET /health) "
              f"using {qa.kb_path}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    question = args.question
    if question is None:
        if sys.stdin.isatty():
            p.error("provide a question argument or JSON on stdin")
        raw = sys.stdin.read()
        try:
            question = json.loads(raw)["question"]
        except (ValueError, KeyError, TypeError):
            question = raw
    print(json.dumps(qa.ask(question), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
