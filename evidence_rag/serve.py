"""Local web interface for a single recommendation request."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evidence_rag.generate import answer

STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[serve] {self.address_string()} {fmt % args}")

    def _send(self, code: int, payload: dict | None = None, body: bytes | None = None, content_type: str = "application/json") -> None:
        data = body if body is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            page = (STATIC / "index.html").read_bytes()
            self._send(200, body=page, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/recommend":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "Request body must be JSON."})
            return
        problem = str(payload.get("problem") or "").strip()
        if len(problem) < 20:
            self._send(400, {"error": "Describe the classification problem in at least a sentence."})
            return
        generator = payload.get("generator") or "extractive"
        if generator not in {"extractive", "local_lm"}:
            self._send(400, {"error": "generator must be extractive or local_lm."})
            return
        result = answer(
            problem=problem,
            constraints=payload.get("constraints") or {},
            task=str(payload.get("task") or "tabular_classification"),
            generator=generator,
            use_retrieval=bool(payload.get("use_retrieval", True)),
        )
        self._send(200, result)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the recommendation prototype")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
