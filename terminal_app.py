from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import enum
import http.server
import json
import shlex
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TextIO

import numpy as np

from actions import Action, ActionType
from ap_types import Axis
from app_runner import AppRunner
from config import AppConfig


DEFAULT_IMAGES_DIR = Path("terminal_output") / "images"
DEFAULT_LIVE_VIEW_PORT = 8765
DEFAULT_LIVE_VIEW_FPS = 2.0
LIVE_VIEW_CONSOLE_LIMIT = 200


class CommandError(ValueError):
    """Error de uso recuperable dentro de la consola."""


class _LiveViewHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], live_view: "LiveView") -> None:
        self.live_view = live_view
        super().__init__(address, _LiveViewHandler)


class _LiveViewHandler(http.server.BaseHTTPRequestHandler):
    """Loopback-only MJPEG endpoint used while the CLI remains interactive."""

    server: _LiveViewHTTPServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_bytes(self, data: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        live_view = self.server.live_view
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            html = """<!doctype html>
<html lang=\"es\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Astropanoptes - vista en vivo</title>
<style>
*{box-sizing:border-box} body{margin:0;background:#080b10;color:#e9eef6;font:16px system-ui,sans-serif}
main{display:grid;place-items:start center;min-height:100vh;padding:16px}
section{width:min(100%,1200px)} h1{font-size:18px;font-weight:650;margin:0 0 10px}
.live{display:block;width:100%;height:min(64vh,720px);object-fit:contain;background:#000;border-radius:8px 8px 0 0}
.console{height:250px;background:#0d121a;border:1px solid #263140;border-top:0;border-radius:0 0 8px 8px;overflow:hidden}
.console-head{display:flex;align-items:center;justify-content:space-between;height:38px;padding:0 12px;border-bottom:1px solid #263140;color:#cbd6e5;font-size:13px;font-weight:650}
.console-head span{color:#718198;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.08em}
#console-lines{height:211px;overflow:auto;padding:8px 12px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.line{display:grid;grid-template-columns:74px 18px 1fr;gap:6px;padding:2px 0}.time{color:#65758b}.mark{font-weight:700}
.command{color:#eef5ff}.command .mark{color:#64d3ff}.output{color:#aebcd0}.output .mark{color:#78d89a}.error{color:#ff9e9e}.error .mark{color:#ff6868}
p{margin:8px 0 0;color:#9eabbc;font-size:13px}
@media(max-height:700px){.live{height:52vh}.console{height:220px}#console-lines{height:181px}}
</style></head><body><main><section><h1>Vista en vivo del telescopio</h1>
<img class=\"live\" src=\"/stream.mjpg\" alt=\"Vista de la camara\">
<div class=\"console\" aria-label=\"Instrucciones enviadas por consola\">
<div class=\"console-head\">Instrucciones por consola <span>solo lectura</span></div>
<div id=\"console-lines\" role=\"log\" aria-live=\"polite\"></div></div>
<p>Los comandos y sus respuestas aparecen aqui automaticamente.</p>
</section></main><script>
const lines=document.getElementById('console-lines');let lastSeq=-1;
function render(items){const follow=lines.scrollHeight-lines.scrollTop-lines.clientHeight<32;lines.replaceChildren();
for(const item of items){const row=document.createElement('div');row.className='line '+item.kind;
const t=document.createElement('span');t.className='time';t.textContent=item.time;
const m=document.createElement('span');m.className='mark';m.textContent=item.kind==='command'?'›':item.kind==='error'?'!':'‹';
const x=document.createElement('span');x.textContent=item.text;row.append(t,m,x);lines.append(row)}
if(follow)lines.scrollTop=lines.scrollHeight}
async function refresh(){try{const r=await fetch('/console.json',{cache:'no-store'});if(!r.ok)return;const p=await r.json();if(p.seq!==lastSeq){lastSeq=p.seq;render(p.entries)}}catch(_){}}
refresh();setInterval(refresh,500);
</script></body></html>""".encode("utf-8")
            self._send_bytes(html, content_type="text/html; charset=utf-8")
            return
        if path == "/console.json":
            seq, entries = live_view.console_snapshot()
            payload = json.dumps(
                {"seq": seq, "entries": entries},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_bytes(payload, content_type="application/json; charset=utf-8")
            return
        if path == "/latest.jpg":
            _seq, frame = live_view.frame_snapshot()
            if not frame:
                self._send_bytes(b"camera frame unavailable\n", content_type="text/plain", status=503)
                return
            self._send_bytes(frame, content_type="image/jpeg")
            return
        if path != "/stream.mjpg":
            self._send_bytes(b"not found\n", content_type="text/plain", status=404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        last_seq = -1
        try:
            while live_view.running:
                seq, frame = live_view.wait_for_frame(after_seq=last_seq, timeout_s=2.0)
                if not frame or seq == last_seq:
                    continue
                last_seq = seq
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


class LiveView:
    """Mirror the live preview to disk and a loopback-only MJPEG viewer."""

    def __init__(self, runner: AppRunner, images_dir: Path) -> None:
        self.runner = runner
        self.images_dir = Path(images_dir)
        self.latest_path = self.images_dir / "live-latest.jpg"
        self.running = False
        self.fps = DEFAULT_LIVE_VIEW_FPS
        self.port = 0
        self._condition = threading.Condition()
        self._frame = b""
        self._seq = 0
        self._console_lock = threading.Lock()
        self._console_entries: collections.deque[dict[str, str]] = collections.deque(
            maxlen=LIVE_VIEW_CONSOLE_LIMIT
        )
        self._console_seq = 0
        self._stop_event = threading.Event()
        self._pump_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._httpd: Optional[_LiveViewHTTPServer] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{int(self.port)}/"

    def start(self, *, port: int = DEFAULT_LIVE_VIEW_PORT, fps: float = DEFAULT_LIVE_VIEW_FPS) -> None:
        if self.running:
            raise CommandError(f"el visor ya esta activo en {self.url}")
        port = int(port)
        fps = float(fps)
        if not (0 <= port <= 65535):
            raise CommandError("el puerto debe estar entre 0 y 65535")
        if not np.isfinite(fps) or not (0.1 <= fps <= 30.0):
            raise CommandError("los FPS del visor deben estar entre 0.1 y 30")

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._stop_event.clear()
        try:
            httpd = _LiveViewHTTPServer(("127.0.0.1", port), self)
        except OSError as exc:
            raise CommandError(f"no se pudo iniciar el visor en el puerto {port}: {exc}") from exc
        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self.running = True
        self._pump_thread = threading.Thread(target=self._pump, name="TerminalLiveViewFrames", daemon=True)
        self._http_thread = threading.Thread(target=httpd.serve_forever, name="TerminalLiveViewHTTP", daemon=True)
        self._pump_thread.start()
        self._http_thread.start()

    def stop(self) -> None:
        if not self.running and self._httpd is None:
            return
        self.running = False
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        httpd = self._httpd
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        for thread in (self._pump_thread, self._http_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._pump_thread = None
        self._http_thread = None
        self._httpd = None

    def frame_snapshot(self) -> tuple[int, bytes]:
        with self._condition:
            return int(self._seq), bytes(self._frame)

    def append_console_entry(self, kind: str, text: str) -> None:
        kind_value = kind if kind in {"command", "output", "error"} else "output"
        clean_text = str(text).strip()
        if not clean_text:
            return
        # Keep the viewer responsive even when a diagnostic command prints a
        # large JSON state dump. The full output remains available in the CLI.
        if len(clean_text) > 4000:
            clean_text = clean_text[:3999] + "…"
        entry = {
            "time": dt.datetime.now().astimezone().strftime("%H:%M:%S"),
            "kind": kind_value,
            "text": clean_text,
        }
        with self._console_lock:
            self._console_entries.append(entry)
            self._console_seq += 1

    def console_snapshot(self) -> tuple[int, list[dict[str, str]]]:
        with self._console_lock:
            return int(self._console_seq), [dict(entry) for entry in self._console_entries]

    def wait_for_frame(self, *, after_seq: int, timeout_s: float) -> tuple[int, bytes]:
        with self._condition:
            if self.running and self._seq <= int(after_seq):
                self._condition.wait(timeout=max(0.0, float(timeout_s)))
            return int(self._seq), bytes(self._frame)

    def _pump(self) -> None:
        period_s = 1.0 / max(0.1, float(self.fps))
        while not self._stop_event.is_set():
            frame = self.runner.get_latest_preview_jpeg()
            if frame:
                data = bytes(frame)
                with self._condition:
                    changed = data != self._frame
                    if changed:
                        self._frame = data
                        self._seq += 1
                        self._condition.notify_all()
                if changed:
                    tmp_path = self.latest_path.with_suffix(".tmp")
                    try:
                        tmp_path.write_bytes(data)
                        tmp_path.replace(self.latest_path)
                    except OSError:
                        # A viewer failure must not interfere with mount/camera control.
                        pass
            self._stop_event.wait(period_s)


def _jsonable(value: Any) -> Any:
    """Convierte estado/config a JSON sin volcar imágenes o arrays grandes."""
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"available": bool(value), "bytes": len(value)}
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "on", "yes", "si", "sí"}:
        return True
    if lowered in {"false", "off", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_assignments(tokens: Iterable[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise CommandError(f"se esperaba nombre=valor; recibido: {token!r}")
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise CommandError("el nombre del parámetro no puede estar vacío")
        params[key] = _parse_value(raw_value)
    return params


def _nested_value(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not part:
            raise CommandError(f"ruta de estado inválida: {path!r}")
        if dataclasses.is_dataclass(current) or hasattr(current, part):
            if not hasattr(current, part):
                raise CommandError(f"no existe el estado {path!r}")
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise CommandError(f"no existe el estado {path!r}")
    if isinstance(current, enum.Enum):
        return current.value
    return current


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if isinstance(actual, bool) and isinstance(expected, str):
        expected = _parse_value(expected)
    elif isinstance(actual, int) and not isinstance(actual, bool):
        try:
            expected = int(expected)
        except (TypeError, ValueError):
            pass
    elif isinstance(actual, float):
        try:
            expected = float(expected)
        except (TypeError, ValueError):
            pass

    if operator in {"=", "=="}:
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == "contains":
        return expected in actual
    raise CommandError(f"operador no soportado: {operator!r}")


HELP_TEXT = """Comandos principales:
  status [--json]                  Estado resumido o JSON completo
  get RUTA                         Lee un campo puntual (ej: get tracking.error_px)
  config [sección]                 Configuración efectiva en JSON
  health                           Errores y operaciones activas
  wait RUTA [OP] VALOR [SEG]       Espera un estado (ej: wait camera.connected true 5)
  await platesolving|goto|mount|record [SEG]
                                    Espera la operación encolada, sin carrera de inicio
  sleep SEG                        Pausa útil en scripts de diagnóstico
  stop                             Parada de emergencia: cancela workers, tracking y montura

  demo on|off
  camera connect [ÍNDICE] | disconnect | set NOMBRE VALOR
  camera record [SEG] [CARPETA] [NOMBRE]
  mount connect [PUERTO] [BAUD] | disconnect | stop | sync
  mount move az|alt DIRECCIÓN PASOS [DELAY_US] [smooth|direct]
  tracking start|stop | set nombre=valor ...
  stacking start|stop|reset | set nombre=valor ...
  stacking save [CARPETA] [NOMBRE] [FORMATO]
  solve OBJETIVO                   Plate solve por nombre o coordenadas
  solve altaz AZ ALT
  platesolving set nombre=valor ... | download [RADIO_DEG]
  goto OBJETIVO | goto altaz AZ ALT | goto cancel|fit|samples|prune|restore|reset
  goto autocalibrate [nombre=valor ...] | goto roll
  overlay sep|expected on|off [nombre=valor ...]

  image live|stack|platesolve [NOMBRE] [ESPERA_SEG]
                                    Guarda JPEG + JSON en la carpeta de imágenes
  view start [PUERTO] [FPS] [no-open]
  view status|open|stop             Visor local en vivo; actualiza live-latest.jpg
  action TIPO [JSON]               Inyecta una Action avanzada para depuración
  help | quit

Los comandos `set` aceptan booleanos, números, null y JSON. Los logs del runtime
van a stderr; las respuestas y el JSON de la CLI van a stdout.
"""


class TerminalApp:
    def __init__(
        self,
        runner: AppRunner,
        *,
        images_dir: Path | str = DEFAULT_IMAGES_DIR,
        output: TextIO = sys.stdout,
        error_output: TextIO = sys.stderr,
        browser_open: Callable[[str], Any] = webbrowser.open,
    ) -> None:
        self.runner = runner
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.output = output
        self.error_output = error_output
        self._browser_open = browser_open
        self.had_error = False
        self._pending_operations: dict[str, int] = {}
        self._live_view = LiveView(runner, self.images_dir)

    def _print(self, message: str = "") -> None:
        self.output.write(message + "\n")
        self.output.flush()
        self._live_view.append_console_entry("output", message)

    def _error(self, message: str) -> None:
        self.had_error = True
        self.error_output.write(f"CLI ERROR: {message}\n")
        self.error_output.flush()
        self._live_view.append_console_entry("error", message)

    def _state_json(self) -> dict[str, Any]:
        return _jsonable(self.runner.get_state())

    def _status(self, *, as_json: bool) -> None:
        state = self.runner.get_state()
        if as_json:
            self._print(json.dumps(_jsonable(state), ensure_ascii=False, indent=2, sort_keys=True))
            return
        self._print(
            " | ".join(
                [
                    f"camera={state.camera.status.value} fps={state.camera.fps_capture:.1f}",
                    f"mount={state.mount.status.value}",
                    f"tracking={state.tracking.status.value}",
                    f"stacking={state.stacking.status.value} frames={state.stacking.frames_used}",
                    f"solve={state.platesolving.status.value}",
                    f"goto={state.goto.status.value}",
                ]
            )
        )

    def _health(self) -> None:
        state = self.runner.get_state()
        camera_failed = state.camera.status.value == "ERROR"
        mount_failed = state.mount.status.value == "ERROR"
        tracking_failed = state.tracking.status.value == "ERROR"
        stacking_failed = state.stacking.status.value == "ERROR"
        platesolving_failed = state.platesolving.status.value == "FAIL"
        goto_failed = state.goto.status.value in {"FAIL", "CANCELLED"}
        errors = {
            "camera": state.camera.last_error if camera_failed else None,
            "mount": state.mount.last_error if mount_failed else None,
            "tracking": state.tracking.last_error if tracking_failed else None,
            "stacking": state.stacking.last_error if stacking_failed else None,
            "platesolving": state.platesolving.reason if platesolving_failed else None,
            "goto": state.goto.reason if goto_failed else None,
        }
        active = {
            "tracking": bool(state.tracking.enabled),
            "stacking": bool(state.stacking.enabled),
            "platesolving": bool(state.platesolving.busy),
            "goto": bool(state.goto.busy),
        }
        self._print(json.dumps({"active": active, "errors": errors}, ensure_ascii=False, indent=2, sort_keys=True))

    def _wait(self, args: list[str]) -> None:
        if len(args) < 2:
            raise CommandError("uso: wait RUTA [==|!=|>|>=|<|<=|contains] VALOR [TIMEOUT_SEG]")
        path = args[0]
        operators = {"=", "==", "!=", ">", ">=", "<", "<=", "contains"}
        if len(args) >= 3 and args[1] in operators:
            operator = args[1]
            expected_raw = args[2]
            timeout_s = float(args[3]) if len(args) >= 4 else 10.0
        else:
            operator = "=="
            expected_raw = args[1]
            timeout_s = float(args[2]) if len(args) >= 3 else 10.0
        expected = _parse_value(expected_raw)
        deadline = time.monotonic() + max(0.0, timeout_s)
        last_value: Any = None
        while True:
            last_value = _nested_value(self.runner.get_state(), path)
            try:
                matched = _compare(last_value, operator, expected)
            except TypeError as exc:
                raise CommandError(f"comparación inválida para {path}: {exc}") from exc
            if matched:
                self._print(f"OK {path}={last_value!r}")
                return
            if time.monotonic() >= deadline:
                raise CommandError(
                    f"timeout esperando {path} {operator} {expected!r}; valor actual={last_value!r}"
                )
            time.sleep(0.05)

    def _operation_counters(self) -> dict[str, dict[str, int]]:
        getter = getattr(self.runner, "get_operation_counters", None)
        if getter is None or not callable(getter):
            raise CommandError("el runner no expone contadores de operaciones")
        return dict(getter())

    def _mark_operation_pending(self, name: str) -> None:
        counters = self._operation_counters()
        if name not in counters:
            raise CommandError(f"operación desconocida: {name!r}")
        self._pending_operations[name] = int(counters[name]["started"])

    def _await_operation(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: await platesolving|goto|mount|record [TIMEOUT_SEG]")
        aliases = {
            "solve": "platesolving",
            "platesolve": "platesolving",
            "mount": "mount_move",
            "move": "mount_move",
            "record": "camera_record",
            "raw": "camera_record",
        }
        name = aliases.get(args[0].lower(), args[0].lower())
        if len(args) >= 2:
            timeout_s = float(args[1])
        elif name == "platesolving":
            timeout_s = float(
                getattr(self.runner.cfg.platesolving, "total_timeout_s", 120.0)
            ) + 15.0
        else:
            timeout_s = 120.0
        if name not in self._pending_operations:
            raise CommandError(f"no hay una operación {name!r} pendiente desde esta consola")
        baseline = int(self._pending_operations[name])
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            counters = self._operation_counters()
            values = counters.get(name)
            if values is None:
                raise CommandError(f"operación desconocida: {name!r}")
            started = int(values["started"])
            finished = int(values["finished"])
            if started > baseline and finished >= started:
                if name == "platesolving":
                    # A successful explicit solve queues automatic sample
                    # validation in the runner. Do not let scripts observe the
                    # transient SOLVING state after `await platesolving`.
                    while time.monotonic() < deadline:
                        sample_reason = getattr(
                            self.runner.get_state().goto,
                            "sample_last_reason",
                            None,
                        )
                        if sample_reason != "SOLVING":
                            break
                        time.sleep(0.01)
                del self._pending_operations[name]
                state = self.runner.get_state()
                section_name = {
                    "mount_move": "mount",
                    "camera_record": "camera",
                }.get(name, name)
                section = getattr(state, section_name)
                payload = {
                    "operation": name,
                    "started": started,
                    "finished": finished,
                    "status": _jsonable(getattr(section, "status", None)),
                    "reason": _jsonable(getattr(section, "reason", None)),
                }
                self._print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                return
            if time.monotonic() >= deadline:
                if name == "platesolving":
                    cancel = getattr(self.runner, "cancel_platesolving", None)
                    if callable(cancel):
                        cancel()
                raise CommandError(
                    f"timeout esperando operación {name}; cancelacion solicitada; "
                    f"started={started} finished={finished} baseline={baseline}"
                )
            time.sleep(0.05)

    def _emergency_stop(self) -> None:
        # MOUNT_STOP is the runner's atomic emergency action: it stops the
        # firmware, cancels GoTo, disables tracking and invalidates sync when
        # a partial move makes the physical step count unknowable. Enqueuing
        # extra cancel actions afterwards could overwrite that diagnostic.
        self.runner.request_mount_stop()

    def _image_bytes(self, source: str) -> Optional[bytes]:
        source = source.lower()
        if source == "live":
            return self.runner.get_latest_preview_jpeg()
        state = self.runner.get_state()
        if source == "stack":
            return state.stacking.preview_jpeg
        if source in {"platesolve", "plate", "solve"}:
            return state.platesolving.debug_jpeg
        raise CommandError("fuente inválida; use live, stack o platesolve")

    def _save_image(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: image live|stack|platesolve [NOMBRE] [ESPERA_SEG]")
        source = args[0].lower()
        filename = args[1] if len(args) >= 2 else ""
        timeout_s = float(args[2]) if len(args) >= 3 else 5.0
        deadline = time.monotonic() + max(0.0, timeout_s)
        data = self._image_bytes(source)
        while not data and time.monotonic() < deadline:
            time.sleep(0.05)
            data = self._image_bytes(source)
        if not data:
            raise CommandError(f"no hay imagen {source!r} disponible después de {timeout_s:g} s")

        self.images_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            safe_name = Path(filename).name
            if not safe_name.lower().endswith((".jpg", ".jpeg")):
                safe_name += ".jpg"
        else:
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            safe_name = f"{source}-{stamp}.jpg"
        image_path = self.images_dir / safe_name
        image_path.write_bytes(data)

        metadata = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "image": str(image_path),
            "source": source,
            "state": self._state_json(),
        }
        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._print(str(image_path))

    def _view(self, args: list[str]) -> None:
        operation = args[0].lower() if args else "status"
        if operation == "start":
            port = int(args[1]) if len(args) >= 2 else DEFAULT_LIVE_VIEW_PORT
            fps = float(args[2]) if len(args) >= 3 else DEFAULT_LIVE_VIEW_FPS
            no_open = len(args) >= 4 and args[3].lower() in {"no-open", "noopen"}
            if len(args) > 4 or (len(args) >= 4 and not no_open):
                raise CommandError("uso: view start [PUERTO] [FPS] [no-open]")
            self._live_view.start(port=port, fps=fps)
            opened = False
            if not no_open:
                try:
                    opened = bool(self._browser_open(self._live_view.url))
                except Exception:
                    opened = False
            self._print(
                json.dumps(
                    {
                        "ok": True,
                        "url": self._live_view.url,
                        "latest_image": str(self._live_view.latest_path),
                        "fps": float(self._live_view.fps),
                        "browser_opened": bool(opened),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return
        if operation == "stop":
            self._live_view.stop()
            self._print("OK")
            return
        if operation == "open":
            if not self._live_view.running:
                raise CommandError("el visor no esta activo; use view start")
            opened = bool(self._browser_open(self._live_view.url))
            self._print(json.dumps({"ok": opened, "url": self._live_view.url}, sort_keys=True))
            return
        if operation == "status" and len(args) <= 1:
            self._print(
                json.dumps(
                    {
                        "running": bool(self._live_view.running),
                        "url": self._live_view.url if self._live_view.running else None,
                        "latest_image": str(self._live_view.latest_path),
                        "latest_available": bool(self._live_view.latest_path.exists()),
                        "fps": float(self._live_view.fps),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return
        raise CommandError("uso: view start [PUERTO] [FPS] [no-open] | status | open | stop")

    def close(self) -> None:
        self._live_view.stop()

    @staticmethod
    def _target(args: list[str]) -> Any:
        if not args:
            raise CommandError("falta el objetivo")
        if args[0].lower() == "altaz":
            if len(args) != 3:
                raise CommandError("uso: ... altaz AZ_DEG ALT_DEG")
            return {"az_deg": float(args[1]), "alt_deg": float(args[2])}
        return " ".join(args)

    def _camera(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: camera connect|disconnect|set|record ...")
        operation = args[0].lower()
        if operation == "connect":
            index = int(args[1]) if len(args) >= 2 else int(self.runner.cfg.camera.camera_index)
            self.runner.request_camera_connect(index)
        elif operation == "disconnect":
            self.runner.request_camera_disconnect()
        elif operation == "set" and len(args) == 3:
            self.runner.request_camera_param(args[1], _parse_value(args[2]))
        elif operation == "record":
            duration_s = float(args[1]) if len(args) >= 2 else 20.0
            out_dir = args[2] if len(args) >= 3 else "raw_output"
            basename = args[3] if len(args) >= 4 else None
            self._mark_operation_pending("camera_record")
            self.runner.request_camera_record_raw(
                duration_s=duration_s,
                out_dir=out_dir,
                basename=basename,
            )
        else:
            raise CommandError("uso: camera connect [ÍNDICE] | disconnect | set NOMBRE VALOR | record [SEG] [DIR] [NOMBRE]")
        self._print("OK")

    def _mount(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: mount connect|disconnect|move|stop|sync ...")
        operation = args[0].lower()
        if operation == "connect":
            port = args[1] if len(args) >= 2 else str(self.runner.cfg.mount.port)
            baud = int(args[2]) if len(args) >= 3 else int(self.runner.cfg.mount.baudrate)
            self.runner.request_mount_connect(port, baud)
        elif operation == "disconnect":
            self.runner.request_mount_disconnect()
        elif operation == "move" and 4 <= len(args) <= 6:
            axis = Axis(args[1].lower())
            direction = int(args[2])
            steps = int(args[3])
            default_delay = (
                self.runner.cfg.mount.slew_delay_us_az
                if axis == Axis.AZ
                else self.runner.cfg.mount.slew_delay_us_alt
            )
            delay_us = int(default_delay)
            profile = "smooth"
            if len(args) >= 5:
                if args[4].lower() in {"smooth", "direct"}:
                    profile = args[4].lower()
                else:
                    delay_us = int(args[4])
            if len(args) == 6:
                profile = args[5].lower()
            if profile not in {"smooth", "direct"}:
                raise CommandError("perfil debe ser smooth o direct")
            self._mark_operation_pending("mount_move")
            self.runner.request_mount_move_steps(
                axis,
                direction,
                steps,
                delay_us,
                profile,
            )
        elif operation == "stop":
            self._emergency_stop()
        elif operation == "sync":
            self.runner.request_mount_sync()
        else:
            raise CommandError(
                "uso: mount connect [PUERTO] [BAUD] | disconnect | "
                "move az|alt +/-1 PASOS [DELAY_US] [smooth|direct] | stop | sync"
            )
        self._print("OK")

    def _tracking(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: tracking start|stop|set ...")
        operation = args[0].lower()
        if operation == "start":
            self.runner.request_tracking_start()
        elif operation == "stop":
            self.runner.request_tracking_stop()
        elif operation == "set" and len(args) >= 2:
            self.runner.request_tracking_params(**_parse_assignments(args[1:]))
        else:
            raise CommandError("uso: tracking start | stop | set nombre=valor ...")
        self._print("OK")

    def _stacking(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: stacking start|stop|reset|set|save ...")
        operation = args[0].lower()
        if operation == "start":
            self.runner.request_stacking_start()
        elif operation == "stop":
            self.runner.request_stacking_stop()
        elif operation == "reset":
            self.runner.request_stacking_reset()
        elif operation == "set" and len(args) >= 2:
            self.runner.request_stacking_params(**_parse_assignments(args[1:]))
        elif operation == "save":
            out_dir = args[1] if len(args) >= 2 else "stack_output"
            basename = args[2] if len(args) >= 3 else "stack"
            fmt = args[3] if len(args) >= 4 else "png"
            self.runner.request_stacking_save(out_dir=out_dir, basename=basename, fmt=fmt)
        else:
            raise CommandError("uso: stacking start|stop|reset | set nombre=valor ... | save [DIR] [NOMBRE] [FMT]")
        self._print("OK")

    def _platesolving(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: platesolving set ... | download [RADIO_DEG]")
        operation = args[0].lower()
        if operation == "set" and len(args) >= 2:
            self.runner.request_platesolving_params(**_parse_assignments(args[1:]))
        elif operation == "download":
            radius = float(args[1]) if len(args) >= 2 else None
            self.runner.request_platesolving_download_current_field(radius)
        else:
            raise CommandError("uso: platesolving set nombre=valor ... | download [RADIO_DEG]")
        self._print("OK")

    def _goto(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: goto OBJETIVO | goto cancel|autocalibrate|roll|fit|samples|prune|restore|reset")
        operation = args[0].lower()
        if operation == "cancel":
            self._emergency_stop()
        elif operation == "autocalibrate":
            self._mark_operation_pending("goto")
            self.runner.request_goto_autocalibrate(_parse_assignments(args[1:]))
        elif operation == "roll":
            self._mark_operation_pending("goto")
            self.runner.request_goto_estimate_roll(_parse_assignments(args[1:]))
        elif operation == "fit":
            self._mark_operation_pending("goto")
            self.runner.request_goto_fit_model(_parse_assignments(args[1:]))
        elif operation == "samples":
            self._mark_operation_pending("goto")
            self.runner.request_goto_list_samples(_parse_assignments(args[1:]))
        elif operation == "prune":
            self._mark_operation_pending("goto")
            self.runner.request_goto_prune_outliers(_parse_assignments(args[1:]))
        elif operation == "restore":
            self._mark_operation_pending("goto")
            self.runner.request_goto_restore_last_log()
        elif operation == "reset":
            self._mark_operation_pending("goto")
            self.runner.request_goto_reset()
        else:
            self._mark_operation_pending("goto")
            self.runner.request_mount_goto(self._target(args))
        self._print("OK")

    def _overlay(self, args: list[str]) -> None:
        if len(args) < 2:
            raise CommandError("uso: overlay sep|expected on|off [nombre=valor ...]")
        kind = args[0].lower()
        enabled = _parse_value(args[1])
        if not isinstance(enabled, bool):
            raise CommandError("el overlay debe indicarse como on u off")
        params = _parse_assignments(args[2:])
        params["enabled"] = enabled
        if kind == "sep":
            self.runner.request_live_sep_params(**params)
        elif kind == "expected":
            self.runner.request_expected_stars_params(**params)
        else:
            raise CommandError("overlay inválido; use sep o expected")
        self._print("OK")

    def _advanced_action(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: action ACTION_TYPE [JSON_OBJECT]")
        try:
            action_type = ActionType(args[0].upper())
        except ValueError as exc:
            raise CommandError(f"ActionType desconocido: {args[0]!r}") from exc
        payload: Any = {}
        if len(args) >= 2:
            try:
                payload = json.loads(" ".join(args[1:]))
            except json.JSONDecodeError as exc:
                raise CommandError(f"payload JSON inválido: {exc}") from exc
        if not isinstance(payload, dict):
            raise CommandError("el payload debe ser un objeto JSON")
        self.runner.enqueue(Action(action_type, payload, time.perf_counter()))
        self._print("OK")

    def execute_line(self, line: str) -> bool:
        """Ejecuta una línea. Retorna False solamente para salir de la consola."""
        clean_line = line.strip()
        if clean_line:
            self._live_view.append_console_entry("command", clean_line)
        try:
            # Preserve JSON quotes for the advanced action command. shlex is
            # ideal for ordinary commands, but intentionally removes those
            # quotes and would turn {"name":"gain"} into {name:gain}.
            raw_parts = line.strip().split(None, 2)
            if raw_parts and raw_parts[0].lower() == "action":
                action_args = raw_parts[1:]
                self._advanced_action(action_args)
                return True
            tokens = shlex.split(line, comments=True)
            if not tokens:
                return True
            command, args = tokens[0].lower(), tokens[1:]
            if command in {"quit", "exit", "salir"}:
                return False
            if command in {"help", "ayuda", "?"}:
                self._print(HELP_TEXT.rstrip())
            elif command in {"status", "state", "estado"}:
                self._status(as_json="--json" in args or "json" in args)
            elif command == "get":
                if len(args) != 1:
                    raise CommandError("uso: get RUTA_DE_ESTADO")
                value = _nested_value(self.runner.get_state(), args[0])
                self._print(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True))
            elif command == "config":
                config_value: Any = self.runner.cfg
                if args:
                    section = args[0]
                    if not hasattr(config_value, section):
                        raise CommandError(f"sección de configuración desconocida: {section!r}")
                    config_value = getattr(config_value, section)
                self._print(json.dumps(_jsonable(config_value), ensure_ascii=False, indent=2, sort_keys=True))
            elif command == "health":
                self._health()
            elif command == "wait":
                self._wait(args)
            elif command == "await":
                self._await_operation(args)
            elif command == "sleep":
                if len(args) != 1:
                    raise CommandError("uso: sleep SEGUNDOS")
                time.sleep(max(0.0, float(args[0])))
                self._print("OK")
            elif command == "demo":
                if len(args) != 1 or not isinstance(_parse_value(args[0]), bool):
                    raise CommandError("uso: demo on|off")
                self.runner.set_simulation_enabled(bool(_parse_value(args[0])))
                self._print("OK")
            elif command == "camera":
                self._camera(args)
            elif command == "mount":
                self._mount(args)
            elif command in {"stop", "emergency-stop", "estop"}:
                self._emergency_stop()
                self._print("OK emergency stop requested")
            elif command == "tracking":
                self._tracking(args)
            elif command == "stacking":
                self._stacking(args)
            elif command == "solve":
                self._mark_operation_pending("platesolving")
                self.runner.request_platesolving_run(self._target(args))
                self._print("OK")
            elif command in {"platesolving", "platesolve"}:
                self._platesolving(args)
            elif command == "goto":
                self._goto(args)
            elif command == "overlay":
                self._overlay(args)
            elif command in {"image", "imagen", "snapshot"}:
                self._save_image(args)
            elif command in {"view", "live-view", "visor"}:
                self._view(args)
            elif command == "action":
                self._advanced_action(args)
            else:
                raise CommandError(f"comando desconocido: {command!r}; use 'help'")
            return True
        except (CommandError, ValueError, TypeError, OSError) as exc:
            self._error(str(exc))
            return True

    def run_interactive(self) -> None:
        self._print("Astropanoptes terminal. Escriba 'help' para ver comandos.")
        self._print(f"Imágenes: {self.images_dir}")
        while True:
            try:
                line = input("astropanoptes> ")
            except EOFError:
                self._print()
                break
            except KeyboardInterrupt:
                self._print("\nParada de emergencia solicitada. Use 'quit' para salir.")
                self._emergency_stop()
                continue
            if not self.execute_line(line):
                break

    def run_script(self, lines: Iterable[str]) -> bool:
        for line in lines:
            if not self.execute_line(line):
                break
            if self.had_error:
                return False
        return not self.had_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py --cli",
        description="Control interactivo y automatizable de Astropanoptes.",
    )
    parser.add_argument("-c", "--command", action="append", default=[], help="comando a ejecutar; se puede repetir")
    parser.add_argument("--script", type=Path, help="archivo con un comando por línea")
    parser.add_argument("--demo", action="store_true", help="inicia con los backends simulados")
    parser.add_argument("--seed", type=int, help="semilla reproducible para --demo")
    parser.add_argument("--connect", action="store_true", help="solicita conexión de cámara y montura al iniciar")
    parser.add_argument("--stay-open", action="store_true", help="abre la consola después de ejecutar comandos/scripts")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="carpeta para imágenes de diagnóstico")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = AppConfig()
    cfg.simulation.enabled = bool(args.demo)
    if args.seed is not None:
        cfg.simulation.seed = int(args.seed)
    runner = AppRunner(cfg, out_log=sys.stderr)
    terminal = TerminalApp(runner, images_dir=args.images_dir)
    runner.start()
    ok = True
    try:
        if args.connect:
            terminal.execute_line("camera connect")
            terminal.execute_line("mount connect")
        if args.command:
            ok = terminal.run_script(args.command) and ok
        if args.script is not None and ok:
            with args.script.expanduser().open("r", encoding="utf-8") as handle:
                ok = terminal.run_script(handle) and ok
        should_interact = (not args.command and args.script is None) or args.stay_open
        if should_interact:
            terminal.run_interactive()
            ok = not terminal.had_error
    except KeyboardInterrupt:
        terminal._print("\nInterrumpido; cerrando de forma segura.")
        ok = False
    except OSError as exc:
        terminal._error(str(exc))
        ok = False
    finally:
        terminal.close()
        runner.stop()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
