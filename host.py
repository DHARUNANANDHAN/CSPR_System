import json
import subprocess
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
STITCH_DIR = BASE_DIR / "stitch"
TEMPLATE_PATH = STITCH_DIR / "dashboard.html"

SCRIPTS = {
    "predictor": {
        "title": "Stampede Predictor",
        "path": BASE_DIR / "predictor.py",
        "category": "Simulation",
        "description": "Generates a live stampede risk map and order-versus-risk plot from a synthetic crowd model.",
        "launchable": True,
    },
    "route": {
        "title": "Rescue Route Planner",
        "path": BASE_DIR / "route.py",
        "category": "Routing",
        "description": "Lets you place start and victim points inside the synthetic crowd and computes the best crowd-aware rescue path.",
        "launchable": True,
    },
    "agent": {
        "title": "Dynamic Rescue Agent",
        "path": BASE_DIR / "agent.py",
        "category": "Routing",
        "description": "Runs a moving rescuer agent that continuously reroutes as the crowd state changes.",
        "launchable": True,
    },
    "cam_pred": {
        "title": "Camera-Based Predictor",
        "path": BASE_DIR / "cam_pred.py",
        "category": "Vision",
        "description": "Analyzes live camera stream and predicts crowd risk with routing overlays.",
        "launchable": True,
    },
}

def read_dashboard() -> bytes:
    return TEMPLATE_PATH.read_bytes()


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class RescueDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in {"/", "/index.html"}:
            self._send_html(read_dashboard())
            return

        if parsed.path == "/api/scripts":
            self._send_json(
                {
                    "scripts": [
                        {
                            "id": key,
                            "title": value["title"],
                            "category": value["category"],
                            "description": value["description"],
                            "path": str(value["path"].name),
                            "launchable": value["launchable"],
                        }
                        for key, value in SCRIPTS.items()
                    ]
                }
            )
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/launch":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload."}, status=HTTPStatus.BAD_REQUEST)
            return

        script_id = payload.get("name", "")
        if script_id not in SCRIPTS:
            self._send_json({"error": "Unknown script."}, status=HTTPStatus.NOT_FOUND)
            return

        script = SCRIPTS[script_id]
        script_path = script["path"]
        if not script_path.exists():
            self._send_json(
                {"error": f"Script file not found: {script_path.name}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        try:
            creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                creationflags=creation_flags,
            )
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to launch {script['title']}: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {
                "ok": True,
                "message": f"{script['title']} launched successfully. Check the opened Python window for output.",
            }
        )

    def _send_html(self, content: bytes, status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        content = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    handler = partial(RescueDashboardHandler, directory=str(BASE_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
