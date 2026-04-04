from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class ControlStateServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._state = {
            "debug_mode": True,
            "mouse_enabled": False,
            "running": False,
        }

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        if self._server is not None:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            def _set_headers(self, status_code=HTTPStatus.OK, content_type="application/json"):
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                self._set_headers(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def do_GET(self):
                if self.path != "/state":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = server.get_state()
                data = json.dumps(payload).encode("utf-8")
                self._set_headers(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if self.path != "/control":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return

                updated = server.update_state(payload)
                data = json.dumps(updated).encode("utf-8")
                self._set_headers(HTTPStatus.OK)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def update_state(self, payload):
        with self._lock:
            if "debug_mode" in payload:
                self._state["debug_mode"] = bool(payload["debug_mode"])
            if "mouse_enabled" in payload:
                self._state["mouse_enabled"] = bool(payload["mouse_enabled"])
            if "running" in payload:
                self._state["running"] = bool(payload["running"])
            return dict(self._state)
