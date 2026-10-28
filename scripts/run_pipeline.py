"""Run the portable workflow directly from an unpacked release."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinsuk_microbiome.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
