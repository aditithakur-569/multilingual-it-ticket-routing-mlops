"""Check tracked notebook code without executing notebook cells."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_python(source, label):
    try:
        compile(source, label, "exec")
        return
    except SyntaxError:
        lines = source.splitlines(keepends=True)
        commands = [
            (index, line.strip())
            for index, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not commands:
            raise

        index, first_line = commands[0]

        if first_line.startswith("%%writefile"):
            parts = shlex.split(first_line)
            if (
                index != 0
                or len(parts) != 2
                or parts[0] != "%%writefile"
                or not parts[1].endswith(".py")
            ):
                raise ValueError(
                    "Expected %%writefile followed by a Python file path."
                )
            lines[index] = "\n"

        elif first_line.startswith("%pip"):
            parts = shlex.split(first_line)
            if (
                len(commands) != 1
                or len(parts) < 3
                or parts[:2] != ["%pip", "install"]
            ):
                raise ValueError("Keep %pip install in its own code cell.")
            lines[index] = "pass\n"

        elif first_line.startswith(("%", "!")):
            raise ValueError(f"Unsupported notebook command: {first_line}")

        else:
            raise

        compile("".join(lines), label, "exec")


def check_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("Expected a notebook containing a cells list.")

    errors = []
    code_count = 0

    for number, cell in enumerate(notebook["cells"], start=1):
        if not isinstance(cell, dict):
            raise ValueError(f"Cell {number} is not a cell object.")
        if cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError(f"Cell {number} has an invalid cell type.")
        if cell["cell_type"] != "code":
            continue

        code_count += 1
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str):
            errors.append(f"Cell {number}: code must be text.")
            continue

        try:
            check_python(source, f"{path.name}, cell {number}")
        except SyntaxError as error:
            errors.append(f"Cell {number}, line {error.lineno}: {error.msg}")
        except ValueError as error:
            errors.append(f"Cell {number}: {error}")

    return code_count, errors


def main():
    result = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.ipynb"],
        cwd=ROOT,
    )
    notebooks = [name for name in result.decode("utf-8").split("\0") if name]

    if not notebooks:
        print("No tracked notebooks found.")
        return 1

    failed = False
    for name in notebooks:
        try:
            count, errors = check_notebook(ROOT / name)
        except (OSError, UnicodeError, ValueError) as error:
            count, errors = 0, [str(error)]

        if errors:
            failed = True
            print(f"FAIL: {name}")
            for error in errors:
                print(f"  {error}")
        else:
            print(f"OK: {name} ({count} code cells)")

    if failed:
        print("Notebook syntax checks failed.")
        return 1

    print(f"Notebook syntax checks passed: {len(notebooks)} notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
