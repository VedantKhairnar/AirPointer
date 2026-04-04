from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np


class PreviewStreamServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._condition = threading.Condition()
        self._frame_jpeg: Optional[bytes] = None
        self._running = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/stream"

    def start(self):
        if self._running:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlparse(self.path).path

                if path in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body style='margin:0;background:#050816;display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;font-family:sans-serif;'>"
                        b"<div>AirPointer preview stream is active.</div></body></html>"
                    )
                    return

                if path == "/frame":
                    frame = server.get_frame_jpeg(block=True, timeout=1.0)
                    if frame is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    return

                if path != "/stream":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while server._running:
                    frame = server.get_frame_jpeg(block=True)
                    if frame is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except BrokenPipeError:
                        break
                    except ConnectionResetError:
                        break

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def update_frame(self, frame_bgr: np.ndarray):
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return
        with self._condition:
            self._frame_jpeg = encoded.tobytes()
            self._condition.notify_all()

    def get_frame_jpeg(self, block: bool = False, timeout: float = 0.5) -> Optional[bytes]:
        with self._condition:
            if not block:
                return self._frame_jpeg
            if self._frame_jpeg is None:
                self._condition.wait(timeout=timeout)
            return self._frame_jpeg
