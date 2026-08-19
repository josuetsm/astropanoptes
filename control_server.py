# control_server.py
"""Servidor de control local: expone la sesión ya abierta de la app por CLI.

Permite que un cliente externo (``python app.py attach``, o cualquier script)
envíe los mismos comandos que ``terminal_app.TerminalApp`` entiende, pero
dirigidos al ``AppRunner`` que ya está corriendo dentro del proceso de la GUI.
Así, mientras la ventana PyQt6 está abierta, se puede intervenir desde un
terminal (por ejemplo, por Codex/Claude) y los cambios se ven en vivo en la
interfaz, y viceversa: lo que el usuario hace en la GUI es visible para quien
está conectado por CLI.

El socket es de dominio Unix, en una ruta por-usuario, y queda restringido a
0600 (solo el dueño puede conectarse).
"""
from __future__ import annotations

import os
import socket
import socketserver
import threading
from pathlib import Path
from typing import Optional

from app_runner import AppRunner
from terminal_app import DEFAULT_IMAGES_DIR, TerminalApp


DEFAULT_SOCKET_PATH = Path.home() / ".astropanoptes" / "control.sock"
RESPONSE_TERMINATOR = "##ASTROPANOPTES-CONTROL-END##"


class ControlServerError(RuntimeError):
    """No se pudo iniciar el servidor de control."""


class _SocketTextWriter:
    """Adaptador mínimo tipo TextIO sobre el ``wfile`` binario del socket."""

    def __init__(self, raw) -> None:
        self._raw = raw

    def write(self, text: str) -> int:
        data = str(text).encode("utf-8", errors="replace")
        self._raw.write(data)
        return len(text)

    def flush(self) -> None:
        self._raw.flush()


class _ControlRequestHandler(socketserver.StreamRequestHandler):
    server: "_ControlSocketServer"

    def handle(self) -> None:
        writer = _SocketTextWriter(self.wfile)
        terminal = TerminalApp(
            self.server.runner,
            images_dir=self.server.images_dir,
            output=writer,
            error_output=writer,
        )
        try:
            while True:
                raw_line = self.rfile.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                try:
                    keep_going = terminal.execute_line(line)
                except Exception as exc:  # pragma: no cover - defensive boundary
                    terminal._error(f"error interno de control: {exc}")
                    keep_going = True
                writer.write(RESPONSE_TERMINATOR + "\n")
                writer.flush()
                if not keep_going:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            terminal.close()


class _ControlSocketServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: Path, runner: AppRunner, images_dir: Path) -> None:
        self.runner = runner
        self.images_dir = images_dir
        super().__init__(str(socket_path), _ControlRequestHandler)


def _clear_stale_socket(socket_path: Path) -> None:
    if not socket_path.exists():
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(socket_path))
    except OSError:
        socket_path.unlink(missing_ok=True)
    else:
        raise ControlServerError(
            f"ya hay una app abierta escuchando en {socket_path}; "
            "usa 'python app.py attach' en vez de abrir otra GUI"
        )
    finally:
        probe.close()


class ControlServer:
    """Envuelve un ``AppRunner`` ya en marcha detrás de un socket de control."""

    def __init__(
        self,
        runner: AppRunner,
        *,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        images_dir: Path | str = DEFAULT_IMAGES_DIR,
    ) -> None:
        self.runner = runner
        self.socket_path = Path(socket_path).expanduser()
        self.images_dir = Path(images_dir)
        self._server: Optional[_ControlSocketServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return
        if not hasattr(socket, "AF_UNIX"):
            raise ControlServerError("el control por CLI requiere sockets Unix (no disponible en esta plataforma)")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        _clear_stale_socket(self.socket_path)
        server = _ControlSocketServer(self.socket_path, self.runner, self.images_dir)
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError:
            pass
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="AstroPanoptesControlServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._server = None
        self.socket_path.unlink(missing_ok=True)
