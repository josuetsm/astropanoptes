from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """Inicia la GUI por defecto o la consola con ``--cli``/``cli``."""
    args = list(sys.argv[1:] if argv is None else argv)
    cli_requested = "--cli" in args or (bool(args) and args[0].lower() in {"cli", "terminal"})
    if cli_requested:
        args = [arg for index, arg in enumerate(args) if arg != "--cli" and not (index == 0 and arg.lower() in {"cli", "terminal"})]
        from terminal_app import main as terminal_main

        return int(terminal_main(args))

    from ui.pyqt6_app import main as ui_main

    ui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
