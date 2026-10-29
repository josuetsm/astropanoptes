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
from ap_types import Axis, TrackingStatus
from app_runner import AppRunner
from config import AppConfig
from console_format import code, explain, reason as reason_text, render_value, result_lines, state_word
from logging_utils import log_info, set_log_file
from pointing.kinematics import MountKinematics


DEFAULT_IMAGES_DIR = Path("terminal_output") / "images"
DEFAULT_LIVE_VIEW_PORT = 8765
DEFAULT_LIVE_VIEW_FPS = 2.0
LIVE_VIEW_CONSOLE_LIMIT = 200


class CommandError(ValueError):
    """Error de uso recuperable dentro de la consola."""


def _wrap(text: str, *, indent: int = 2, width: int = 78) -> list[str]:
    """Prosa en lineas de ancho fijo, indentada como el detalle de una respuesta."""
    import textwrap

    pad = " " * indent
    return textwrap.wrap(text, width=width, initial_indent=pad, subsequent_indent=pad)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _int_arg(text: str, name: str) -> int:
    """Un argumento entero, con un error que dice cual y por que.

    Sin esto, `mount move az 1 abc` respondia con el texto interno de Python
    ("invalid literal for int() with base 10"), que no dice que argumento fallo.
    """
    try:
        return int(str(text).strip())
    except ValueError:
        raise CommandError(f"se esperaba un número entero para {name}; recibí {text!r}") from None


def _float_arg(text: str, name: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        raise CommandError(f"se esperaba un número para {name}; recibí {text!r}") from None


def _axis_arg(text: str) -> Axis:
    try:
        return Axis(str(text).strip().lower())
    except ValueError:
        raise CommandError(f"el eje debe ser az o alt; recibí {text!r}") from None


def _direction_arg(text: str) -> int:
    value = _int_arg(text, "la dirección")
    if value not in (-1, 1):
        raise CommandError(f"la dirección debe ser 1 o -1; recibí {text!r}")
    return value


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


def _children(node: Any) -> list[str]:
    if dataclasses.is_dataclass(node):
        return [f.name for f in dataclasses.fields(node)]
    if isinstance(node, dict):
        return [str(k) for k in node]
    return []


def _nested_value(root: Any, path: str) -> Any:
    """Un campo del estado por su ruta con puntos.

    Si la ruta no existe, el error dice que campos hay en ese nivel: sin eso,
    `get stacking.frames` obligaba a pedir el estado entero para descubrir que
    el campo se llama `frames_used`.
    """
    current = root
    walked: list[str] = []
    for part in path.split("."):
        if not part:
            raise CommandError(f"ruta de estado mal formada: {path!r}")
        if dataclasses.is_dataclass(current) and hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            here = ".".join(walked) or "el estado"
            options = _children(current)
            hint = f"; en {here} hay: {', '.join(options)}" if options else ""
            raise CommandError(f"no existe {path}{hint}")
        walked.append(part)
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


HELP_TEXT = """Estado y espera
  status [--json]                    resumen de cada subsistema
  get RUTA                           un campo del estado (get tracking.rate_az)
  config [SECCION]                   configuración efectiva
  health                             errores, operaciones en curso y latencias
  wait RUTA [OP] VALOR [SEG]         espera a que el estado cumpla algo
                                       (wait camera.connected true 5)
  await solve|goto|mount|record|focus [SEG]
                                     espera la última operación que lanzaste
  sleep SEG                          pausa
  stop                               parada de emergencia: montura, goto y tracking

Cámara y montura
  camera connect [INDICE] | disconnect | set NOMBRE VALOR
  camera record [SEG] [DIR] [NOMBRE] | stop
  mount connect [PUERTO] [BAUDIOS] | disconnect | stop | sync
  mount move az|alt 1|-1 PASOS [RETARDO_US] [smooth|direct]
                                     un ciclo de transmisión son 12800 pasos:
                                     8° en azimut, 4° en altura
  demo on|off

Foco
  focus in|out [PASOS] | goto POS|PRESET | home | zero | cancel
  focus auto [PRESET] [nombre=valor ...]
  focus save NOMBRE [POS] | presets | forget NOMBRE | status | set nombre=valor ...

Tracking y apilado
  tracking start | stop | why | set nombre=valor ...
  stacking start | stop | reset | set nombre=valor ... | save [DIR] [NOMBRE] [FMT]
  transmission status | apply        rizado del reductor aprendido del tracking

Plate solving y goto
  solve OBJETIVO | solve altaz AZ ALT
  platesolving source live|stack | set nombre=valor ... | download [RADIO_DEG]
  goto OBJETIVO | goto altaz AZ ALT | goto cancel
  goto autocalibrate [nombre=valor ...]   sitúa la montura por deriva y resuelve;
                                          después hace falta mount sync
  goto roll | fit | samples | prune | restore | reset
  overlay sep|expected on|off [nombre=valor ...]

Imágenes y visor
  image live|stack|platesolve [NOMBRE] [ESPERA_SEG]
  view start [PUERTO] [FPS] [no-open] | status | open | stop

Otros
  action TIPO [JSON]                 inyecta una acción en bruto, para depurar
  help | quit

Cómo responde
  Cada comando produce una línea que empieza por ok o por error, y debajo,
  indentado, el detalle si lo hay. En scripts (-c, --script) cada respuesta
  lleva encima el comando que la produjo, precedido de ›.

  La mayoría de los comandos solo encolan una acción y responden
  "ok  en cola: ...". Que esté en cola no significa que esté hecho: para
  saberlo usa wait o await, o mira el evento que llega cuando termina.

  Las respuestas van por stdout. Los eventos -- lo que pasa mientras tanto --
  van por stderr, con hora, nivel y subsistema en columnas:
    21:54:06.931  error  goto      la montura no está sincronizada

  Con --json, cualquier comando responde solo con el documento JSON.
  Un script se detiene en el primer error y dice en qué comando.
  El log completo de la sesión, con trazas, queda en terminal_output/logs/.
"""


class TerminalApp:
    def __init__(
        self,
        runner: AppRunner,
        *,
        images_dir: Path | str = DEFAULT_IMAGES_DIR,
        output: TextIO = sys.stdout,
        error_output: Optional[TextIO] = None,
        browser_open: Callable[[str], Any] = webbrowser.open,
        echo_commands: bool = False,
    ) -> None:
        self.runner = runner
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.output = output
        # Los errores de un comando son su respuesta, asi que van por el mismo
        # canal que el `ok`: si fueran por stderr, separar los canales dejaria el
        # error lejos del comando que lo produjo. Se acepta un canal propio solo
        # por compatibilidad.
        self.error_output = error_output if error_output is not None else output
        self._browser_open = browser_open
        self.echo_commands = bool(echo_commands)
        self.had_error = False
        self._json = False
        self._pending_operations: dict[str, int] = {}
        self._live_view = LiveView(runner, self.images_dir)

    # ------------------------------------------------------------- salida

    def _write(self, lines: Iterable[str], *, stream: Optional[TextIO] = None, kind: str = "output") -> None:
        target = stream if stream is not None else self.output
        lines = list(lines)
        for line in lines:
            target.write(line + "\n")
        target.flush()
        if lines:
            self._live_view.append_console_entry(kind, "\n".join(lines))

    def _print(self, message: str = "") -> None:
        """Texto tal cual, para bloques de prosa (ayuda, informes)."""
        self._write(message.split("\n") if message else [""])

    def _ok(self, summary: str = "", detail: Any = None) -> None:
        """Respuesta correcta. Con `--json`, solo el documento JSON."""
        if self._json:
            payload = detail if detail is not None else {"ok": True, "summary": summary}
            self._write([json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)])
            return
        self._write(result_lines(True, summary, _jsonable(detail) if detail is not None else None))

    def _queued(self, what: str) -> None:
        """Una orden aceptada, que todavia no se ha cumplido.

        La mayoria de los comandos solo encolan una accion. Decirlo en la
        respuesta evita leer `ok` como "ya esta hecho": para eso estan `wait` y
        `await`, y el evento que llega por stderr cuando termina.
        """
        self._ok(f"en cola: {what}")

    def _error(self, message: str, detail: Any = None) -> None:
        self.had_error = True
        self._write(
            result_lines(False, message, _jsonable(detail) if detail is not None else None),
            stream=self.error_output,
            kind="error",
        )

    def _state_json(self) -> dict[str, Any]:
        return _jsonable(self.runner.get_state())

    def _status(self, *, as_json: bool) -> None:
        state = self.runner.get_state()
        if as_json:
            self._json = True
            self._ok(detail=state)
            return

        def failure(status, reason, explanation_source=None) -> str:
            raw = code(status)
            if raw not in {"error", "fail", "cancelled"}:
                return ""
            why = reason_text(reason) if reason else ""
            text = explain(reason) if reason else ""
            return "  ".join(part for part in (why, text) if part)

        rows = []
        cam = state.camera
        rows.append(("camara", state_word("camara", cam.status),
                     f"{cam.fps_capture:.1f} fps" if cam.connected else "", failure(cam.status, cam.last_error)))
        goto = state.goto
        where = ""
        if goto.pointing_valid:
            where = f"az {goto.pointing_az_deg:.3f}°  alt {goto.pointing_alt_deg:.3f}°"
        sync = "sincronizada" if goto.synced else ("sin sincronizar" if state.mount.connected else "")
        rows.append(("montura", state_word("montura", state.mount.status),
                     "  ".join(x for x in (where, sync) if x), failure(state.mount.status, state.mount.last_error)))
        trk = state.tracking
        rows.append(("tracking", state_word("tracking", trk.status), "", failure(trk.status, trk.last_error)))
        stk = state.stacking
        rows.append(("apilado", state_word("apilado", stk.status),
                     _count(int(stk.frames_used), "cuadro", "cuadros") if stk.enabled or stk.frames_used else "",
                     failure(stk.status, stk.last_error)))
        ps = state.platesolving
        solved = ""
        if code(ps.status) == "ok":
            solved = (f"ra {ps.center_ra_deg:.4f}°  dec {ps.center_dec_deg:.4f}°  "
                      f"{int(ps.n_inliers)} inliers")
        rows.append(("solve", state_word("solve", ps.status), solved, failure(ps.status, ps.reason)))
        rows.append(("goto", state_word("goto", goto.status),
                     _count(int(goto.manual_samples), "muestra", "muestras") if goto.manual_samples else "",
                     failure(goto.status, goto.reason)))

        name_width = max(len(r[0]) for r in rows) + 2
        width = max(len(r[1]) for r in rows)
        lines = []
        for name, word, extra, why in rows:
            line = f"  {name:<{name_width}}{word:<{width}}"
            tail = "  ".join(x for x in (extra, why) if x)
            lines.append((line + "  " + tail).rstrip())
        self._write(["ok"] + lines)

    def _health(self) -> None:
        state = self.runner.get_state()
        failed = {
            "camara": (state.camera.status, state.camera.last_error),
            "montura": (state.mount.status, state.mount.last_error),
            "tracking": (state.tracking.status, state.tracking.last_error),
            "apilado": (state.stacking.status, state.stacking.last_error),
            "solve": (state.platesolving.status, state.platesolving.reason),
            "goto": (state.goto.status, state.goto.reason),
        }
        errors = {
            name: (reason_text(reason) if reason else code(status))
            for name, (status, reason) in failed.items()
            if code(status) in {"error", "fail", "cancelled"}
        }
        active = [
            name for name, busy in (
                ("tracking", state.tracking.enabled),
                ("apilado", state.stacking.enabled),
                ("solve", state.platesolving.busy),
                ("goto", state.goto.busy),
            ) if busy
        ]
        getter = getattr(self.runner, "get_performance_metrics", None)
        performance = getter() if callable(getter) else None
        if self._json:
            self._ok(detail={"active": active, "errors": errors, "performance": performance})
            return

        summary = []
        summary.append(f"{len(errors)} con error" if errors else "sin errores")
        summary.append("en curso: " + ", ".join(active) if active else "nada en curso")
        lines = ["ok  " + "; ".join(summary)]
        for name, why in errors.items():
            text = explain(why)
            lines.append(f"  error     {name}: {why}" + (f"  {text}" if text else ""))
        sections = (performance or {}).get("sections", {}) if isinstance(performance, dict) else {}
        if sections:
            lines.append(f"  latencia por ciclo (ms), {int((performance or {}).get('sample_count', 0))} muestras")
            width = max(len(k) for k in sections)
            for key, stats in sections.items():
                if not isinstance(stats, dict):
                    continue
                cells = "  ".join(
                    f"{q} {float(stats[q]):<8.3g}" for q in ("p50", "p95", "p99", "max") if q in stats
                ).rstrip()
                lines.append(f"    {key.removesuffix('_ms'):<{width}}  {cells}")
        self._write(lines)

    def _wait(self, args: list[str]) -> None:
        if len(args) < 2:
            raise CommandError("uso: wait RUTA [==|!=|>|>=|<|<=|contains] VALOR [TIMEOUT_SEG]")
        path = args[0]
        operators = {"=", "==", "!=", ">", ">=", "<", "<=", "contains"}
        if len(args) >= 3 and args[1] in operators:
            operator = args[1]
            expected_raw = args[2]
            timeout_s = _float_arg(args[3], "el timeout") if len(args) >= 4 else 10.0
        else:
            operator = "=="
            expected_raw = args[1]
            timeout_s = _float_arg(args[2], "el timeout") if len(args) >= 3 else 10.0
        expected = _parse_value(expected_raw)
        started = time.monotonic()
        deadline = started + max(0.0, timeout_s)
        last_value: Any = None
        while True:
            last_value = _nested_value(self.runner.get_state(), path)
            try:
                matched = _compare(last_value, operator, expected)
            except TypeError as exc:
                raise CommandError(f"no se puede comparar {path} con {expected_raw!r}: {exc}") from exc
            if matched:
                self._ok(f"{path} = {render_value(_jsonable(last_value))}  ({time.monotonic() - started:.1f} s)")
                return
            if time.monotonic() >= deadline:
                raise CommandError(
                    f"{path} no llegó a {operator} {render_value(expected)} en {timeout_s:g} s; "
                    f"vale {render_value(_jsonable(last_value))}"
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
            raise CommandError("uso: await platesolving|goto|mount|record|focus [TIMEOUT_SEG]")
        aliases = {
            "solve": "platesolving",
            "platesolve": "platesolving",
            "mount": "mount_move",
            "move": "mount_move",
            "record": "camera_record",
            "raw": "camera_record",
            "focus": "focuser",
            "autofocus": "focuser",
        }
        name = aliases.get(args[0].lower(), args[0].lower())
        if len(args) >= 2:
            timeout_s = _float_arg(args[1], "el timeout")
        elif name == "platesolving":
            timeout_s = float(getattr(self.runner.cfg.platesolving, "total_timeout_s", 120.0)) + 15.0
        else:
            timeout_s = 120.0
        if name not in self._pending_operations:
            raise CommandError(
                f"no hay ninguna operación {name} lanzada desde esta consola; "
                "await espera la última que lanzaste tú"
            )
        baseline = int(self._pending_operations[name])
        began = time.monotonic()
        deadline = began + max(0.0, timeout_s)
        while True:
            counters = self._operation_counters()
            values = counters.get(name)
            if values is None:
                raise CommandError(f"operación desconocida: {name}")
            started = int(values["started"])
            finished = int(values["finished"])
            if started > baseline and finished >= started:
                if name == "platesolving":
                    # Un solve explicito encola la validacion de la muestra en el
                    # runner; no dejar que el script vea el estado SOLVING de paso.
                    while time.monotonic() < deadline:
                        if getattr(self.runner.get_state().goto, "sample_last_reason", None) != "SOLVING":
                            break
                        time.sleep(0.01)
                del self._pending_operations[name]
                self._report_operation(name, time.monotonic() - began)
                return
            if time.monotonic() >= deadline:
                if name == "platesolving":
                    cancel = getattr(self.runner, "cancel_platesolving", None)
                    if callable(cancel):
                        cancel()
                raise CommandError(
                    f"{name} no terminó en {timeout_s:g} s"
                    + ("; se pidió cancelarla" if name == "platesolving" else "")
                )
            time.sleep(0.05)

    def _report_operation(self, name: str, elapsed_s: float) -> None:
        """Como acabo una operacion, con los numeros que importan de cada una.

        Antes devolvia `"ok": true, "status": "FAIL"`: ok significaba que la
        espera habia funcionado, no la operacion. Ahora el resultado es el de la
        operacion, y un fallo detiene un script igual que cualquier otro error.
        """
        state = self.runner.get_state()
        section = {
            "mount_move": state.mount,
            "camera_record": state.camera,
            "focuser": state.focuser,
        }.get(name, getattr(state, name, None))
        status = getattr(section, "status", None)
        reason = getattr(section, "reason", None) or getattr(section, "last_error", None)
        failed = code(status) in {"fail", "error", "cancelled"}
        label = {"platesolving": "solve", "mount_move": "movimiento", "camera_record": "grabación",
                 "focuser": "foco"}.get(name, name)

        detail: dict[str, Any] = {}
        if name == "platesolving":
            ps = state.platesolving
            if not failed:
                detail = {
                    "ra_deg": round(float(ps.center_ra_deg), 5),
                    "dec_deg": round(float(ps.center_dec_deg), 5),
                    "inliers": int(ps.n_inliers),
                    "rms_px": round(float(ps.rms_px), 3),
                    "rotation_deg": round(float(ps.theta_deg), 2),
                }
        elif name == "goto":
            goto = state.goto
            detail = {"synced": bool(goto.synced), "samples": int(goto.manual_samples)}
            if goto.pointing_valid:
                detail["az_deg"] = round(float(goto.pointing_az_deg), 4)
                detail["alt_deg"] = round(float(goto.pointing_alt_deg), 4)
        elif name == "focuser":
            detail = {"position": state.focuser.position, "best_position": state.focuser.best_position}
        if reason:
            detail["reason"] = reason_text(reason)
        if failed and explain(reason):
            detail["explanation"] = explain(reason)

        took = f"{elapsed_s:.1f} s"
        if failed:
            why = reason_text(reason) if reason else code(status)
            self._error(f"{label} falló: {why.removeprefix('err_')}  ({took})", detail or None)
            return
        self._ok(f"{label} terminó  ({took})", detail or None)

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
        raise CommandError(f"fuente desconocida {source!r}; usa live, stack o platesolve")

    def _save_image(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: image live|stack|platesolve [NOMBRE] [ESPERA_SEG]")
        source = args[0].lower()
        filename = args[1] if len(args) >= 2 else ""
        timeout_s = _float_arg(args[2], "la espera") if len(args) >= 3 else 5.0
        deadline = time.monotonic() + max(0.0, timeout_s)
        data = self._image_bytes(source)
        while not data and time.monotonic() < deadline:
            time.sleep(0.05)
            data = self._image_bytes(source)
        if not data:
            raise CommandError(f"no hay imagen {source} disponible tras esperar {timeout_s:g} s")

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
        self._ok(str(image_path), {"image": str(image_path), "metadata": str(metadata_path), "bytes": len(data)})

    def _view(self, args: list[str]) -> None:
        operation = args[0].lower() if args else "status"
        if operation == "start":
            port = _int_arg(args[1], "el puerto") if len(args) >= 2 else DEFAULT_LIVE_VIEW_PORT
            fps = _float_arg(args[2], "los fps") if len(args) >= 3 else DEFAULT_LIVE_VIEW_FPS
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
            self._ok(
                f"visor en {self._live_view.url}",
                {
                    "url": self._live_view.url,
                    "latest_image": str(self._live_view.latest_path),
                    "fps": float(self._live_view.fps),
                    "browser_opened": bool(opened),
                },
            )
            return
        if operation == "stop":
            self._live_view.stop()
            self._ok("visor detenido")
            return
        if operation == "open":
            if not self._live_view.running:
                raise CommandError("el visor no está activo; arráncalo con view start")
            opened = bool(self._browser_open(self._live_view.url))
            if not opened:
                raise CommandError(f"no se pudo abrir el navegador; la dirección es {self._live_view.url}")
            self._ok(f"abierto {self._live_view.url}")
            return
        if operation == "status" and len(args) <= 1:
            running = bool(self._live_view.running)
            self._ok(
                f"visor en {self._live_view.url}" if running else "visor detenido",
                {
                    "running": running,
                    "url": self._live_view.url if running else None,
                    "latest_image": str(self._live_view.latest_path),
                    "latest_available": bool(self._live_view.latest_path.exists()),
                    "fps": float(self._live_view.fps),
                },
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
            return {"az_deg": _float_arg(args[1], "el azimut"), "alt_deg": _float_arg(args[2], "la altura")}
        return " ".join(args)

    def _camera(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: camera connect|disconnect|set|record ...")
        operation = args[0].lower()
        if operation == "connect":
            index = _int_arg(args[1], "el índice") if len(args) >= 2 else int(self.runner.cfg.camera.camera_index)
            self.runner.request_camera_connect(index)
            self._queued(f"conectar cámara {index}")
        elif operation == "disconnect":
            self.runner.request_camera_disconnect()
            self._queued("desconectar cámara")
        elif operation == "set" and len(args) == 3:
            value = _parse_value(args[2])
            self.runner.request_camera_param(args[1], value)
            self._queued(f"cámara {args[1]} = {render_value(value)}")
        elif operation == "record":
            duration_s = _float_arg(args[1], "la duración") if len(args) >= 2 else 20.0
            out_dir = args[2] if len(args) >= 3 else "raw_output"
            basename = args[3] if len(args) >= 4 else None
            self._mark_operation_pending("camera_record")
            self.runner.request_camera_record_raw(
                duration_s=duration_s,
                out_dir=out_dir,
                basename=basename,
            )
            self._queued(f"grabar RAW {duration_s:g} s en {out_dir}  (await record)")
        elif operation == "stop":
            self.runner.request_camera_stop_record_raw()
            self._queued("parar la grabación")
        else:
            raise CommandError("uso: camera connect [ÍNDICE] | disconnect | set NOMBRE VALOR | record [SEG] [DIR] [NOMBRE] | stop")

    def _mount(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: mount connect|disconnect|move|stop|sync ...")
        operation = args[0].lower()
        if operation == "connect":
            port = args[1] if len(args) >= 2 else str(self.runner.cfg.mount.port)
            baud = _int_arg(args[2], "los baudios") if len(args) >= 3 else int(self.runner.cfg.mount.baudrate)
            self.runner.request_mount_connect(port, baud)
            self._queued(f"conectar montura en {port} a {baud} baudios")
            return
        elif operation == "disconnect":
            self.runner.request_mount_disconnect()
            self._queued("desconectar montura")
            return
        elif operation == "move" and 4 <= len(args) <= 6:
            axis = _axis_arg(args[1])
            direction = _direction_arg(args[2])
            steps = _int_arg(args[3], "los pasos")
            if steps <= 0:
                raise CommandError(f"los pasos deben ser positivos; recibí {steps}. El sentido va en la dirección (1 o -1)")
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
                    delay_us = _int_arg(args[4], "el retardo en µs")
            if len(args) == 6:
                profile = args[5].lower()
            if profile not in {"smooth", "direct"}:
                raise CommandError(f"el perfil debe ser smooth o direct; recibí {profile!r}")
            self._mark_operation_pending("mount_move")
            self.runner.request_mount_move_steps(
                axis,
                direction,
                steps,
                delay_us,
                profile,
            )
            # Los grados ayudan a no confundir pasos con angulo: en altitud un
            # ciclo de transmision (12800 pasos) son 4 grados, en azimut 8.
            degrees = steps * abs(MountKinematics().deg_per_step(axis))
            self._queued(
                f"mover {axis.value} {'+' if direction > 0 else '-'}{steps} pasos "
                f"({degrees:.3f}°), {profile}, {delay_us} µs  (await mount)"
            )
        elif operation == "stop":
            self._emergency_stop()
            self._ok("parada de emergencia: montura, goto y tracking detenidos")
        elif operation == "sync":
            self.runner.request_mount_sync()
            self._queued("sincronizar la montura con el último solve")
        else:
            raise CommandError(
                "uso: mount connect [PUERTO] [BAUD] | disconnect | "
                "move az|alt 1|-1 PASOS [DELAY_US] [smooth|direct] | stop | sync"
            )

    def _focus(self, args: list[str]) -> None:
        if not args:
            raise CommandError(
                "uso: focus in|out [PASOS] | auto [...] | cancel | home | "
                "goto POS|PRESET | save NOMBRE [POS] | presets | forget NOMBRE | "
                "zero | status | set nombre=valor ..."
            )
        operation = args[0].lower()
        cfg = self.runner.cfg.focuser

        if operation in {"in", "acercar", "near"}:
            steps = _int_arg(args[1], "los pasos") if len(args) >= 2 else int(cfg.step_size)
            self._mark_operation_pending("focuser")
            self.runner.request_focuser_move(+1, steps)
            self._queued(f"acercar el foco {steps} pasos  (await focus)")
        elif operation in {"out", "alejar", "far"}:
            steps = _int_arg(args[1], "los pasos") if len(args) >= 2 else int(cfg.step_size)
            self._mark_operation_pending("focuser")
            self.runner.request_focuser_move(-1, steps)
            self._queued(f"alejar el foco {steps} pasos  (await focus)")
        elif operation in {"auto", "autofocus", "buscar"}:
            # "focus auto x2" fija el preset; sin nombre se deduce del preset
            # mas cercano a la posicion actual.
            rest = list(args[1:])
            if rest and "=" not in rest[0]:
                params = _parse_assignments(rest[1:])
                params["preset"] = rest[0]
            else:
                params = _parse_assignments(rest)
            self._mark_operation_pending("focuser")
            self.runner.request_focuser_autofocus(params)
            where = f" desde el preset {params['preset']}" if "preset" in params else ""
            self._queued(f"buscar foco{where}  (await focus)")
        elif operation in {"cancel", "cancelar"}:
            self.runner.request_focuser_cancel()
            self._queued("cancelar el enfoque")
        elif operation == "goto":
            if len(args) != 2:
                raise CommandError("uso: focus goto POSICION|NOMBRE_DE_PRESET")
            self._mark_operation_pending("focuser")
            try:
                position = int(args[1])
            except ValueError:
                # No es un numero: se interpreta como preset, que es como se
                # escribe en la practica ("focus goto x2").
                self.runner.request_focuser_preset(args[1])
                self._queued(f"llevar el foco al preset {args[1]}  (await focus)")
            else:
                self.runner.request_focuser_goto(position)
                self._queued(f"llevar el foco a {position:+d}  (await focus)")
        elif operation in {"zero", "cero"}:
            self.runner.request_focuser_zero()
            self._queued("fijar el cero de sesión del foco aquí (no es homing)")
        elif operation in {"home", "homing"}:
            self._mark_operation_pending("focuser")
            self.runner.request_focuser_home()
            self._queued("homing del foco contra la zona de patinaje  (await focus)")
        elif operation in {"save", "guardar"}:
            if not (2 <= len(args) <= 3):
                raise CommandError("uso: focus save NOMBRE [POSICION]")
            position = _int_arg(args[2], "la posición") if len(args) == 3 else None
            self.runner.request_focuser_save_preset(args[1], position)
            where = f"{position:+d}" if position is not None else "la posición actual"
            self._queued(f"guardar el preset {args[1]} en {where}")
        elif operation in {"forget", "olvidar", "borrar"}:
            if len(args) != 2:
                raise CommandError("uso: focus forget NOMBRE")
            self.runner.request_focuser_delete_preset(args[1])
            self._queued(f"olvidar el preset {args[1]}")
        elif operation in {"presets", "lista"}:
            presets = self.runner.get_focus_presets()
            if not presets:
                self._ok("no hay presets de foco guardados")
                return
            homed = bool(self.runner.get_state().focuser.homed)
            from focuser import FocusPresets

            lines = [f"ok  {len(presets)} preset(s)"]
            for name in sorted(presets):
                entry = presets[name]
                origin = "con homing" if entry["homed"] else "sin homing"
                center, spread = FocusPresets.prior(entry)
                n = len(entry.get("history", []))
                if spread is None:
                    learned = f"{n} medida(s)" if n else "sin historial"
                else:
                    learned = f"{n} noches, centro {center:+d}, dispersión ±{spread:.0f}"
                mark = "" if bool(entry["homed"]) == homed else "  (origen distinto al actual)"
                lines.append(f"  {name:<8} {int(entry['position']):+8d}  {origin}  {learned}{mark}")
            self._write(lines)
            return
        elif operation in {"load", "usar", "aplicar"}:
            if len(args) != 2:
                raise CommandError("uso: focus load NOMBRE")
            self._mark_operation_pending("focuser")
            self.runner.request_focuser_preset(args[1])
            self._queued(f"llevar el foco al preset {args[1]}  (await focus)")
        elif operation in {"status", "estado"}:
            focuser = self.runner.get_state().focuser
            summary = (
                f"posición {focuser.position:+d}" if focuser.position is not None else "posición desconocida"
            ) + (", moviéndose" if focuser.moving else "") + (", con homing" if focuser.homed else ", sin homing")
            self._ok(summary, focuser)
            return
        elif operation == "set":
            params = _parse_assignments(args[1:])
            if not params:
                raise CommandError("uso: focus set nombre=valor ...")
            self.runner.request_focuser_params(**params)
            self._queued("parámetros del foco: " + ", ".join(f"{k}={render_value(v)}" for k, v in params.items()))
        elif operation in {"reset", "defaults"}:
            self.runner.request_focuser_reset_defaults()
            self._queued("restaurar los parámetros del foco por defecto")
        else:
            raise CommandError(f"operación de foco desconocida: {operation!r}; usa focus sin argumentos para ver la sintaxis")

    def _tracking_calib(self, args: list[str]) -> None:
        if len(args) not in (4, 6):
            raise CommandError(
                "uso: tracking calib A00 A01 A10 A11 [b0 b1]   "
                "(px/s por paso/s; columnas = ejes az, alt)"
            )
        names = ("A00", "A01", "A10", "A11", "b0", "b1")
        values = [_float_arg(v, name) for v, name in zip(args, names)]
        from actions import tracking_set_calib

        self.runner.enqueue(tracking_set_calib(*values))

    def _tracking_why(self) -> None:
        """Por que el tracking no esta moviendo la montura.

        El tracking puede estar encendido y comandando exactamente cero sin que
        nada en la ventana lo delate: el chip queda verde y los motores quietos.
        Esto recorre las compuertas en orden y dice cual es la que falta. El
        veredicto va en la primera linea, que es lo primero que se lee.
        """
        state = self.runner.get_state()
        tracking = state.tracking
        rate_az = float(tracking.rate_az)
        rate_alt = float(tracking.rate_alt)
        moving = max(abs(rate_az), abs(rate_alt)) > 1e-6

        def yes(flag: bool) -> str:
            return "sí" if flag else "no"

        if bool(getattr(tracking, "anchor_lost", False)):
            anchor = "perdida, fuera de campo"
        elif bool(getattr(tracking, "anchor_chained", False)):
            anchor = f"encadenada, a {float(getattr(tracking, 'anchor_px', 0.0)):.1f} px"
        else:
            anchor = "en uso"
        rows = [
            ("montura conectada", yes(state.mount.connected)),
            ("cámara conectada", yes(state.camera.connected)),
            ("tracking activado", f"sí ({state_word('tracking', tracking.status)})" if tracking.enabled else "no"),
            ("medida válida", "sí" if tracking.measurement_valid else f"no  ({reason_text(tracking.measurement_reason)})"),
            ("fuentes detectadas", str(int(tracking.n_det))),
            ("calibración", code(tracking.calib_src)),
            ("referencia original", anchor),
            ("montura sincronizada", "sí" if state.goto.synced else "no  (sin feed-forward)"),
            ("feed-forward listo", yes(tracking.ff_ready)),
            ("velocidad comandada", f"{rate_az:+.4f} / {rate_alt:+.4f} pasos/s"),
        ]

        if not state.mount.connected:
            verdict = "falta conectar la montura"
            advice = ""
        elif not state.camera.connected:
            verdict = "falta conectar la cámara"
            advice = (
                "El tracking es en lazo cerrado: mide la deriva de las estrellas en la "
                "imagen, y sin imagen no tiene de dónde sacar la corrección."
            )
        elif tracking.status == TrackingStatus.ERROR:
            verdict = f"el tracking se detuvo por errores: {tracking.last_error}"
            advice = (
                "Vuelve a activarlo; si se corta de nuevo enseguida, el fallo es "
                "permanente y hay que mirar el log de la sesión."
            )
        elif not tracking.enabled:
            verdict = "el tracking no está activado: usa tracking start"
            advice = ""
        elif moving:
            period = 1.0 / max(abs(rate_az), abs(rate_alt))
            verdict = "está comandando"
            advice = (
                f"A esta velocidad es un microstep cada {period:.1f} s: no se ve ni se oye "
                "moverse. Para comprobarlo mira get tracking.rate_az, no los motores."
            )
        elif not tracking.measurement_valid and not tracking.ff_ready:
            verdict = f"sin medida válida ({reason_text(tracking.measurement_reason)}) y sin feed-forward"
            advice = (
                "Sin ninguna de las dos el tracking comanda cero y no manda ni un MOVE. "
                "Hacen falta estrellas que pueda seguir, o un plate solve que sincronice "
                "la montura."
            )
        elif bool(getattr(tracking, "anchor_lost", False)):
            verdict = "siguiendo, pero la referencia original quedó fuera de campo"
            advice = (
                "Ya no se puede volver a ella: lo que hace ahora es estabilizar el campo "
                "actual, no el objetivo inicial."
            )
        elif not tracking.measurement_valid:
            verdict = f"sin medida ({reason_text(tracking.measurement_reason)}); solo feed-forward sideral"
            advice = ""
        else:
            verdict = "medida válida y velocidad cero: no hay deriva que corregir ahora"
            advice = ""

        width = max(len(label) for label, _ in rows)
        lines = [f"ok  {verdict}"]
        lines += [f"  {label:<{width}}  {value}" for label, value in rows]
        if advice:
            lines += [""] + _wrap(advice, indent=2)
        self._write(lines)

    def _tracking(self, args: list[str]) -> None:
        if args and args[0].lower() in {"why", "porque", "diagnose", "diag"}:
            self._tracking_why()
            return
        if args and args[0].lower() in {"calib", "calibracion"}:
            self._tracking_calib(args[1:])
            self._queued("instalar la calibración de tracking indicada")
            return
        if not args:
            raise CommandError("uso: tracking start|stop|why|set ...")
        operation = args[0].lower()
        if operation == "start":
            self.runner.request_tracking_start()
            self._queued("arrancar el tracking  (tracking why si no se mueve)")
        elif operation == "stop":
            self.runner.request_tracking_stop()
            self._queued("detener el tracking")
        elif operation == "set" and len(args) >= 2:
            params = _parse_assignments(args[1:])
            self.runner.request_tracking_params(**params)
            self._queued("parámetros de tracking: " + ", ".join(f"{k}={render_value(v)}" for k, v in params.items()))
        else:
            raise CommandError("uso: tracking start | stop | why | set nombre=valor ...")

    def _stacking(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: stacking start|stop|reset|set|save ...")
        operation = args[0].lower()
        if operation == "start":
            self.runner.request_stacking_start()
            self._queued("arrancar el apilado")
        elif operation == "stop":
            self.runner.request_stacking_stop()
            self._queued("detener el apilado")
        elif operation == "reset":
            self.runner.request_stacking_reset()
            self._queued("vaciar el apilado")
        elif operation == "set" and len(args) >= 2:
            params = _parse_assignments(args[1:])
            self.runner.request_stacking_params(**params)
            self._queued("parámetros de apilado: " + ", ".join(f"{k}={render_value(v)}" for k, v in params.items()))
        elif operation == "save":
            out_dir = args[1] if len(args) >= 2 else "stack_output"
            basename = args[2] if len(args) >= 3 else "stack"
            fmt = args[3] if len(args) >= 4 else "png"
            self.runner.request_stacking_save(out_dir=out_dir, basename=basename, fmt=fmt)
            self._queued(f"guardar el apilado en {out_dir}/ (raw, pesos y metadatos)")
        else:
            raise CommandError("uso: stacking start|stop|reset | set nombre=valor ... | save [DIR] [NOMBRE] [FMT]")

    def _platesolving(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: platesolving set ... | download [RADIO_DEG]")
        operation = args[0].lower()
        if operation == "source" and len(args) == 2:
            source = str(args[1]).strip().lower()
            if source not in {"live", "stack"}:
                raise CommandError(f"la fuente debe ser live o stack; recibí {source!r}")
            self.runner.request_platesolving_params(source=source)
            self._queued(f"resolver sobre {'el mosaico apilado' if source == 'stack' else 'el cuadro en vivo'}")
        elif operation == "set" and len(args) >= 2:
            params = _parse_assignments(args[1:])
            self.runner.request_platesolving_params(**params)
            self._queued("parámetros del solve: " + ", ".join(f"{k}={render_value(v)}" for k, v in params.items()))
        elif operation == "download":
            radius = _float_arg(args[1], "el radio") if len(args) >= 2 else None
            self.runner.request_platesolving_download_current_field(radius)
            self._queued("descargar el catálogo del campo actual" + (f" en {radius:g}°" if radius else ""))
        else:
            raise CommandError(
                "uso: platesolving set nombre=valor ... | source live|stack | download [RADIO_DEG]"
            )

    def _goto(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: goto OBJETIVO | goto cancel|autocalibrate|roll|fit|samples|prune|restore|reset")
        operation = args[0].lower()
        if operation == "cancel":
            self._emergency_stop()
            self._ok("goto cancelado; la montura queda sin sincronizar si se movió a medias")
            return
        simple = {
            "autocalibrate": (self.runner.request_goto_autocalibrate, True,
                              "situar la montura por deriva y resolver (luego mount sync)"),
            "roll": (self.runner.request_goto_estimate_roll, True, "estimar el giro de la cámara"),
            "fit": (self.runner.request_goto_fit_model, True, "ajustar el modelo de apuntado"),
            "samples": (self.runner.request_goto_list_samples, True, "listar las muestras del modelo"),
            "prune": (self.runner.request_goto_prune_outliers, True, "descartar muestras atípicas"),
            "restore": (self.runner.request_goto_restore_last_log, False, "restaurar el último modelo guardado"),
            "reset": (self.runner.request_goto_reset, False, "borrar el modelo y la sincronización"),
        }
        self._mark_operation_pending("goto")
        if operation in simple:
            request, takes_params, what = simple[operation]
            if takes_params:
                request(_parse_assignments(args[1:]))
            else:
                request()
            self._queued(f"{what}  (await goto)")
            return
        target = self._target(args)
        self.runner.request_mount_goto(target)
        if isinstance(target, dict):
            where = f"az {target['az_deg']:g}°, alt {target['alt_deg']:g}°"
        else:
            where = str(target)
        self._queued(f"goto a {where}  (await goto)")

    def _transmission(self, args: list[str]) -> None:
        operation = args[0].lower() if args else "status"
        if operation == "status":
            getter = getattr(self.runner, "get_transmission_error_report", None)
            if not callable(getter):
                raise CommandError("el runner no expone el aprendizaje de transmisión")
            report = _jsonable(getter())
            coverage = (report or {}).get("coverage", {}) if isinstance(report, dict) else {}
            fitted = bool((report or {}).get("fitted")) if isinstance(report, dict) else False
            summary = (
                "rizado aprendido" if fitted
                else f"aprendiendo: cobertura de fase {render_value(coverage.get('min', 0.0))} (hace falta 0.6)"
            )
            self._ok(summary, report)
            return
        if operation == "apply":
            apply = getattr(self.runner, "apply_learned_transmission_error", None)
            if not callable(apply):
                raise CommandError("el runner no expone el aprendizaje de transmisión")
            if not apply():
                raise CommandError(
                    "cobertura de fase insuficiente: sigue con el tracking activo, "
                    "barre la fase solo; mira transmission status"
                )
            self._ok("rizado de transmisión instalado en el modelo")
            return
        raise CommandError("uso: transmission status | apply")

    def _overlay(self, args: list[str]) -> None:
        if len(args) < 2:
            raise CommandError("uso: overlay sep|expected on|off [nombre=valor ...]")
        kind = args[0].lower()
        enabled = _parse_value(args[1])
        if not isinstance(enabled, bool):
            raise CommandError(f"el overlay se activa con on u off; recibí {args[1]!r}")
        params = _parse_assignments(args[2:])
        params["enabled"] = enabled
        if kind == "sep":
            self.runner.request_live_sep_params(**params)
        elif kind == "expected":
            self.runner.request_expected_stars_params(**params)
        else:
            raise CommandError(f"overlay desconocido {kind!r}; usa sep o expected")
        self._queued(f"overlay {kind} {'activado' if enabled else 'desactivado'}")

    def _advanced_action(self, args: list[str]) -> None:
        if not args:
            raise CommandError("uso: action ACTION_TYPE [JSON_OBJECT]")
        try:
            action_type = ActionType(args[0].upper())
        except ValueError as exc:
            raise CommandError(f"tipo de acción desconocido: {args[0]!r}") from exc
        payload: Any = {}
        if len(args) >= 2:
            try:
                payload = json.loads(" ".join(args[1:]))
            except json.JSONDecodeError as exc:
                raise CommandError(f"el payload no es JSON válido: {exc}") from exc
        if not isinstance(payload, dict):
            raise CommandError("el payload debe ser un objeto JSON")
        self.runner.enqueue(Action(action_type, payload, time.perf_counter()))
        self._queued(f"acción {code(action_type)}")

    def execute_line(self, line: str) -> bool:
        """Ejecuta una linea. Devuelve False solo para salir de la consola."""
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            return True
        self._live_view.append_console_entry("command", clean_line)
        self._json = False
        try:
            # Las comillas del JSON de `action` se conservan: shlex las quitaria
            # y convertiria {"name":"gain"} en {name:gain}.
            raw_parts = clean_line.split(None, 2)
            if raw_parts and raw_parts[0].lower() == "action":
                self._echo(clean_line)
                self._advanced_action(raw_parts[1:])
                return True
            try:
                tokens = shlex.split(clean_line, comments=True)
            except ValueError as exc:
                self._echo(clean_line)
                raise CommandError(f"no se pudo leer la línea: {exc}") from None
            if not tokens:
                return True
            # `--json` vale para cualquier comando: la respuesta pasa a ser solo
            # el documento JSON, para quien prefiera parsear a leer.
            if "--json" in tokens:
                self._json = True
                tokens = [t for t in tokens if t != "--json"]
            if not self._json:
                self._echo(clean_line)
            command, args = tokens[0].lower(), tokens[1:]
            return self._dispatch(command, args)
        except CommandError as exc:
            self._error(str(exc))
        except (ValueError, TypeError) as exc:
            self._error(f"valor inválido: {exc}")
        except OSError as exc:
            self._error(f"error del sistema: {exc}")
        finally:
            self._json = False
        return True

    def _echo(self, line: str) -> None:
        if self.echo_commands:
            self._write([f"› {line}"], kind="command")

    def _dispatch(self, command: str, args: list[str]) -> bool:
        if command in {"quit", "exit", "salir"}:
            return False
        if command in {"help", "ayuda", "?"}:
            self._write(["ok  ayuda"] + ["  " + l if l else "" for l in HELP_TEXT.rstrip().splitlines()])
        elif command in {"status", "state", "estado"}:
            self._status(as_json=self._json or "json" in args)
        elif command == "get":
            if len(args) != 1:
                raise CommandError("uso: get RUTA  (por ejemplo get tracking.rate_az)")
            value = _jsonable(_nested_value(self.runner.get_state(), args[0]))
            if isinstance(value, (dict, list)) and value and not self._json:
                self._ok(args[0], value)
            elif self._json:
                self._ok(detail={"path": args[0], "value": value})
            else:
                self._ok(render_value(value))
        elif command == "config":
            config_value: Any = self.runner.cfg
            if args:
                section = args[0]
                if not hasattr(config_value, section):
                    options = ", ".join(_children(config_value))
                    raise CommandError(f"no existe la sección {section!r}; hay: {options}")
                config_value = getattr(config_value, section)
            self._ok(f"configuración{' de ' + args[0] if args else ''}", config_value)
        elif command == "health":
            self._health()
        elif command == "wait":
            self._wait(args)
        elif command == "await":
            self._await_operation(args)
        elif command == "sleep":
            if len(args) != 1:
                raise CommandError("uso: sleep SEGUNDOS")
            seconds = max(0.0, _float_arg(args[0], "los segundos"))
            time.sleep(seconds)
            self._ok(f"esperé {seconds:g} s")
        elif command == "demo":
            if len(args) != 1 or not isinstance(_parse_value(args[0]), bool):
                raise CommandError("uso: demo on|off")
            enabled = bool(_parse_value(args[0]))
            self.runner.set_simulation_enabled(enabled)
            self._queued(f"modo demo {'activado' if enabled else 'desactivado'}")
        elif command == "camera":
            self._camera(args)
        elif command == "mount":
            self._mount(args)
        elif command in {"focus", "focuser", "foco"}:
            self._focus(args)
        elif command in {"stop", "emergency-stop", "estop"}:
            self._emergency_stop()
            self._ok("parada de emergencia: montura, goto y tracking detenidos")
        elif command == "tracking":
            self._tracking(args)
        elif command == "stacking":
            self._stacking(args)
        elif command == "solve":
            self._mark_operation_pending("platesolving")
            target = self._target(args)
            self.runner.request_platesolving_run(target)
            where = (f"cerca de az {target['az_deg']:g}°, alt {target['alt_deg']:g}°"
                     if isinstance(target, dict) else f"cerca de {target}")
            self._queued(f"resolver {where}  (await solve)")
        elif command in {"platesolving", "platesolve"}:
            self._platesolving(args)
        elif command == "goto":
            self._goto(args)
        elif command == "overlay":
            self._overlay(args)
        elif command in {"transmission", "transmision"}:
            self._transmission(args)
        elif command in {"image", "imagen", "snapshot"}:
            self._save_image(args)
        elif command in {"view", "live-view", "visor"}:
            self._view(args)
        else:
            raise CommandError(f"comando desconocido: {command!r}; usa help para ver la lista")
        return True

    def run_interactive(self) -> None:
        self._write([
            "ok  consola de astropanoptes; escribe help para ver los comandos",
            f"  imágenes  {self.images_dir}",
        ])
        while True:
            try:
                line = input("astropanoptes> ")
            except EOFError:
                self._write([""])
                break
            except KeyboardInterrupt:
                self._emergency_stop()
                self._write(["", "ok  parada de emergencia por Ctrl-C; escribe quit para salir"])
                continue
            if not self.execute_line(line):
                break

    def run_script(self, lines: Iterable[str]) -> bool:
        """Ejecuta comandos en orden y se detiene en el primer error.

        Si se detiene, la ultima linea dice en que comando y cuantos quedaron sin
        ejecutar: sin eso, quien lee la salida no sabe si el script acabo o se
        corto.
        """
        commands = [line.rstrip("\n") for line in lines]
        runnable = [c for c in commands if c.strip() and not c.strip().startswith("#")]
        done = 0
        for line in commands:
            if not line.strip() or line.strip().startswith("#"):
                continue
            keep_going = self.execute_line(line)
            done += 1
            if self.had_error:
                left = len(runnable) - done
                self._write([
                    f"error  script detenido en el comando {done} de {len(runnable)}: {line.strip()}"
                    + (f"; quedaron {left} sin ejecutar" if left else "")
                ], stream=self.error_output, kind="error")
                return False
            if not keep_going:
                break
        return True


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
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = set_log_file(Path("terminal_output") / "logs" / f"sesion-{stamp}.log")
    runner = AppRunner(cfg, out_log=sys.stderr)
    scripted = bool(args.command) or args.script is not None
    terminal = TerminalApp(runner, images_dir=args.images_dir, echo_commands=scripted)
    log_info(None, f"Runner: sesión iniciada{' en modo demo' if args.demo else ''}; log completo en {log_path}")
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
        terminal._error("interrumpido con Ctrl-C; cerrando de forma segura")
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
