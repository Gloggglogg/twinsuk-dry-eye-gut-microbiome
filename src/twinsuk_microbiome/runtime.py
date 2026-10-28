"""Resolve bundled source modules without machine-specific paths."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PACKAGE_ROOT / "analysis_sources"


def analysis_module(name: str):
    """Import original bare module names, preserving their shared globals."""
    if not name.isidentifier():
        raise ValueError("Expected a bundled Python module name")
    expected = ANALYSIS_DIR / f"{name}.py"
    if not expected.is_file():
        raise ImportError(f"Bundled analysis module is missing: {name}")
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() != expected.resolve():
        raise ImportError(f"A different module named {name} is already loaded; use a fresh Python process")
    source_path = str(ANALYSIS_DIR)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected.resolve():
        raise ImportError(f"The requested bundled module was shadowed: {name}")
    return module


@contextmanager
def module_globals(module, **replacements):
    """Redirect explicit artifact readers for one source-function call."""
    original = {key: getattr(module, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(module, key, value)
        yield module
    finally:
        for key, value in original.items():
            setattr(module, key, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints() -> dict[str, str]:
    files = sorted(ANALYSIS_DIR.glob("*.py"))
    return {path.name: sha256(path) for path in files}
