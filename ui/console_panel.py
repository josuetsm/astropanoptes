"""Consola embebida: los mismos comandos de ``terminal_app`` dentro de la GUI.

``TerminalApp`` ya acepta un ``AppRunner`` externo y cualquier ``TextIO`` como
salida, que es justamente lo que hace ``control_server`` sobre un socket. Aquí
se reutiliza esa misma pieza, pero escribiendo en el panel de logs de la
ventana: así los parámetros que no valen un control propio en la UI siguen
estando a mano de noche, sin abrir otra terminal.

Los comandos corren en un hilo aparte porque ``wait``, ``await`` y ``sleep``
bloquean por segundos y congelarían la ventana.
"""
from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app_runner import AppRunner
from terminal_app import DEFAULT_IMAGES_DIR


#: Verbos aceptados por ``TerminalApp.execute_line``, para autocompletar.
CONSOLE_COMMANDS = (
    "action",
    "await",
    "camera",
    "config",
    "demo",
    "get",
    "goto",
    "health",
    "help",
    "image",
    "mount",
    "overlay",
    "platesolving",
    "sleep",
    "solve",
    "stacking",
    "status",
    "stop",
    "tracking",
    "transmission",
    "view",
    "wait",
)

def _monospace_font() -> QFont:
    """Fuente de ancho fijo, para que el JSON de `config`/`status` quede alineado."""
    font = QFont("Menlo")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(12)
    return font


_COLOR_COMMAND = "#7fd1ff"
_COLOR_ERROR = "#ff9a9a"
_MAX_BLOCKS = 3000


class _SignalWriter:
    """Adaptador tipo ``TextIO`` que emite líneas completas por una señal Qt."""

    def __init__(self, emit, kind: str) -> None:
        self._emit = emit
        self._kind = kind
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line, self._kind)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer, self._kind)
            self._buffer = ""


class _ConsoleWorker(QObject):
    """Ejecuta líneas de comando contra el runner ya en marcha, fuera del hilo GUI."""

    output = pyqtSignal(str, str)
    idle = pyqtSignal()

    def __init__(self, runner: AppRunner, images_dir: Path) -> None:
        super().__init__()
        self._runner = runner
        self._images_dir = images_dir
        self._terminal = None

    @pyqtSlot()
    def start(self) -> None:
        from terminal_app import TerminalApp

        self._terminal = TerminalApp(
            self._runner,
            images_dir=self._images_dir,
            output=_SignalWriter(self.output.emit, "out"),
            error_output=_SignalWriter(self.output.emit, "err"),
        )

    @pyqtSlot(str)
    def execute(self, line: str) -> None:
        try:
            if self._terminal is None:
                self.output.emit("consola no inicializada", "err")
                return
            # `quit`/`exit` cerrarían la consola de terminal; aquí no aplica,
            # porque la sesión vive mientras la ventana esté abierta.
            self._terminal.execute_line(line)
        except Exception as exc:  # pragma: no cover - frontera defensiva
            self.output.emit(f"error interno de consola: {exc}", "err")
        finally:
            self.idle.emit()

    @pyqtSlot()
    def shutdown(self) -> None:
        if self._terminal is not None:
            self._terminal.close()
            self._terminal = None


class _CommandLineEdit(QLineEdit):
    """Entrada de comandos con historial navegable por ↑/↓."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._history: list[str] = []
        self._index = 0
        self._draft = ""

    def remember(self, line: str) -> None:
        if line and (not self._history or self._history[-1] != line):
            self._history.append(line)
        self._index = len(self._history)
        self._draft = ""

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Up and self._history:
            if self._index == len(self._history):
                self._draft = self.text()
            self._index = max(0, self._index - 1)
            self.setText(self._history[self._index])
            return
        if key == Qt.Key.Key_Down and self._history:
            self._index = min(len(self._history), self._index + 1)
            self.setText(
                self._draft if self._index == len(self._history) else self._history[self._index]
            )
            return
        super().keyPressEvent(event)


class ConsolePanel(QWidget):
    """Panel de logs con entrada de comandos, sobre la sesión ya abierta."""

    _submitted = pyqtSignal(str)
    _shutdown_requested = pyqtSignal()

    def __init__(
        self,
        runner: AppRunner,
        *,
        images_dir: Path | str = DEFAULT_IMAGES_DIR,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._pending = 0

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(_MAX_BLOCKS)
        self.view.setFont(_monospace_font())
        self.view.setToolTip(
            "Registro de eventos, errores y salida de los comandos ejecutados en la consola."
        )

        self.input = _CommandLineEdit()
        self.input.setPlaceholderText("comando…  (help para ver la lista, ↑/↓ historial)")
        self.input.setFont(_monospace_font())
        self.input.setToolTip(
            "Ejecuta los mismos comandos de 'python app.py --cli' sobre esta sesión, "
            "sin abrir otra terminal. Ej: tracking set sidereal_ff_gain=0.8"
        )
        self.input.returnPressed.connect(self._submit)

        completer = QCompleter(sorted(CONSOLE_COMMANDS), self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.input.setCompleter(completer)

        self.lbl_busy = QLabel("")
        self.lbl_busy.setFixedWidth(18)
        self.lbl_busy.setToolTip("Indica que hay un comando en ejecución.")

        self.btn_help = QPushButton("?")
        self.btn_help.setFixedWidth(30)
        self.btn_help.setToolTip("Muestra la lista de comandos disponibles.")
        self.btn_help.clicked.connect(lambda: self.run_command("help"))

        prompt_row = QHBoxLayout()
        prompt_row.setContentsMargins(0, 0, 0, 0)
        prompt_row.setSpacing(6)
        prompt_row.addWidget(QLabel(">"))
        prompt_row.addWidget(self.input, stretch=1)
        prompt_row.addWidget(self.lbl_busy)
        prompt_row.addWidget(self.btn_help)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(prompt_row)

        self._thread = QThread(self)
        self._thread.setObjectName("AstroPanoptesConsole")
        self._worker = _ConsoleWorker(runner, Path(images_dir))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.output.connect(self._append_worker_output)
        self._worker.idle.connect(self._command_finished)
        self._submitted.connect(self._worker.execute)
        self._shutdown_requested.connect(self._worker.shutdown)
        self._thread.start()

    # -- salida ---------------------------------------------------------

    def append_log(self, message: str) -> None:
        """Agrega una línea de log de la aplicación (mismo rol que el panel viejo)."""
        self._append(message, None)

    def _append(self, message: str, color: Optional[str]) -> None:
        # Siempre se inserta como HTML: `append` solo interpreta marcado cuando
        # el texto empieza por una etiqueta, así que el span va siempre. La
        # sangría se preserva con espacios duros (importa para el JSON de
        # `config`/`status`), pero los espacios interiores se dejan tal cual
        # para que las líneas largas sigan haciendo wrap.
        raw = str(message)
        body = raw.lstrip(" ")
        indent = "&nbsp;" * (len(raw) - len(body))
        style = f"color:{color};" if color else ""
        self.view.append(f'<span style="{style}">{indent}{_html.escape(body)}</span>')

    @pyqtSlot(str, str)
    def _append_worker_output(self, message: str, kind: str) -> None:
        self._append(message, _COLOR_ERROR if kind == "err" else None)

    # -- entrada --------------------------------------------------------

    def _submit(self) -> None:
        line = self.input.text().strip()
        if not line:
            return
        self.input.remember(line)
        self.input.clear()
        self.run_command(line)

    def run_command(self, line: str) -> None:
        """Encola una línea de comando y la refleja en el registro."""
        self._append(f"> {line}", _COLOR_COMMAND)
        self._pending += 1
        self.lbl_busy.setText("⏳")
        self._submitted.emit(line)

    @pyqtSlot()
    def _command_finished(self) -> None:
        self._pending = max(0, self._pending - 1)
        if self._pending == 0:
            self.lbl_busy.setText("")

    # -- ciclo de vida --------------------------------------------------

    def shutdown(self) -> None:
        if not self._thread.isRunning():
            return
        self._shutdown_requested.emit()
        self._thread.quit()
        self._thread.wait(2000)
