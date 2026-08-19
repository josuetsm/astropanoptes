# control_client.py
"""Cliente CLI que se conecta a una GUI de Astropanoptes ya abierta.

A diferencia de ``terminal_app.py`` (que crea su propio ``AppRunner`` y por
lo tanto su propia conexión a cámara/montura), este cliente no toca hardware
directamente: envía comandos por el socket de control que la GUI expone
(``control_server.py``) y muestra las respuestas. Sirve tanto para que el
usuario intervenga manualmente en una sesión abierta como para que un agente
(Codex/Claude) opere esa misma sesión mientras el usuario la observa.
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import IO, Iterable, Optional

from control_server import DEFAULT_SOCKET_PATH, RESPONSE_TERMINATOR


class AttachError(RuntimeError):
    """No se pudo conectar a una app ya abierta."""


def connect(socket_path: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
    except FileNotFoundError as exc:
        raise AttachError(
            f"no hay ninguna app abierta escuchando en {socket_path}; "
            "abre primero la GUI (python app.py)"
        ) from exc
    except ConnectionRefusedError as exc:
        raise AttachError(
            f"el socket {socket_path} existe pero no responde; "
            "cierra y vuelve a abrir la app"
        ) from exc
    return sock


def _send_command(rfile, sock: socket.socket, line: str, *, output: IO[str]) -> bool:
    sock.sendall((line + "\n").encode("utf-8"))
    had_error = False
    while True:
        raw = rfile.readline()
        if not raw:
            raise AttachError("la app cerró la conexión de control inesperadamente")
        text = raw.decode("utf-8", errors="replace").rstrip("\n")
        if text == RESPONSE_TERMINATOR:
            break
        if text.startswith("CLI ERROR:"):
            had_error = True
        print(text, file=output)
    return not had_error


def run_commands(sock: socket.socket, commands: Iterable[str], *, output: IO[str] = sys.stdout) -> bool:
    rfile = sock.makefile("rb")
    ok = True
    for command in commands:
        ok = _send_command(rfile, sock, command, output=output) and ok
    return ok


def run_interactive(sock: socket.socket, *, output: IO[str] = sys.stdout) -> bool:
    rfile = sock.makefile("rb")
    print("Astropanoptes attach. Conectado a la app abierta. Escriba 'help' o 'quit'.", file=output)
    ok = True
    while True:
        try:
            line = input("astropanoptes(attached)> ")
        except EOFError:
            print(file=output)
            break
        except KeyboardInterrupt:
            print(file=output)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        ok = _send_command(rfile, sock, line, output=output) and ok
        if stripped.lower() in {"quit", "exit", "salir"}:
            break
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py attach",
        description="Conecta por CLI a una GUI de Astropanoptes ya abierta, sin abrir otra sesión.",
    )
    parser.add_argument("-c", "--command", action="append", default=[], help="comando a enviar; se puede repetir")
    parser.add_argument("--script", type=Path, help="archivo con un comando por línea")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH, help="ruta del socket de control")
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="abre la consola interactiva después de ejecutar comandos/script",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    socket_path = Path(args.socket).expanduser()
    try:
        sock = connect(socket_path)
    except AttachError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ok = True
    try:
        if args.command:
            ok = run_commands(sock, args.command) and ok
        if args.script is not None:
            with args.script.expanduser().open("r", encoding="utf-8") as handle:
                lines = [raw.rstrip("\n") for raw in handle if raw.strip()]
            ok = run_commands(sock, lines) and ok
        should_interact = (not args.command and args.script is None) or args.stay_open
        if should_interact:
            ok = run_interactive(sock) and ok
    except AttachError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        ok = False
    finally:
        sock.close()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
