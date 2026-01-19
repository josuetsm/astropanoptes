#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_py_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if p.is_file()
        and ".venv" not in p.parts
        and "__pycache__" not in p.parts
    )


def count_patterns(text: str) -> dict[str, int]:
    except_pass_inline = len(re.findall(r"except Exception:\s*pass\b", text))
    except_pass_multiline = len(re.findall(r"except Exception:\s*\n\s*pass\b", text))
    bare_except = len(re.findall(r"^\s*except:\b", text, flags=re.MULTILINE))
    return {
        "except_exception_pass": except_pass_inline + except_pass_multiline,
        "bare_except": bare_except,
    }


def count_prints(text: str) -> int:
    return len(re.findall(r"\bprint\(", text))


def main() -> None:
    files = iter_py_files(ROOT)
    total_except_exception_pass = 0
    total_bare_except = 0
    total_prints = 0
    print_details = []

    for path in files:
        content = path.read_text(encoding="utf-8")
        counts = count_patterns(content)
        total_except_exception_pass += counts["except_exception_pass"]
        total_bare_except += counts["bare_except"]

        if path.name != "logging_utils.py":
            prints_here = count_prints(content)
            total_prints += prints_here
            if prints_here:
                print_details.append((str(path.relative_to(ROOT)), prints_here))

    print("Audit metrics:")
    print(f"- except Exception + pass -> {total_except_exception_pass}")
    print(f"- bare except -> {total_bare_except}")
    print(f"- print(...) outside logging_utils.py -> {total_prints}")
    if print_details:
        print("  Files with print(...) calls:")
        for fname, count in print_details:
            print(f"    - {fname}: {count}")


if __name__ == "__main__":
    main()
