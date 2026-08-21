"""environment version check"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def pinned_requirements(path: Path) -> dict[str, str]:
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"requirement is not exactly pinned: {line}")
        name, wanted = line.split("==", 1)
        pins[name.strip()] = wanted.strip()
    return pins


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = []
    if sys.version_info < (3, 10):
        errors.append(
            f"Python {sys.version.split()[0]} is too old; use Python 3.10 or newer")
    for package, wanted in pinned_requirements(root / "requirements.txt").items():
        try:
            found = version(package)
        except PackageNotFoundError:
            errors.append(f"{package} is missing (required {wanted})")
            continue
        if found != wanted:
            errors.append(f"{package}=={found}, required {package}=={wanted}")
    if errors:
        print("Environment check failed; no analysis has been run:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("Create a clean environment with `PYTHON=python3.10 bash setup.sh`.",
              file=sys.stderr)
        raise SystemExit(2)
    print(f"Environment OK: Python {sys.version.split()[0]} and all pinned packages.")


if __name__ == "__main__":
    main()
