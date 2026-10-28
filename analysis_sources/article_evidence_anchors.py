from __future__ import annotations

import re
from pathlib import Path


TABLE_PREFIXES = {
    "table1_characteristics.tex": "T1",
    "table_s1a_alpha_diversity.tex": "TS1A",
    "table_s1b_severity_whs.tex": "TS1B",
    "table_s2_richness_robustness.tex": "TS2",
    "table_s3_ecological_context.tex": "TS3",
    "table_s4_genus_results.tex": "TS4",
    "table_s5_twin.tex": "TS5",
}

ANCHOR_RE = re.compile(r"^\s*%\s*SRC\s+A3-")


def anchor_generated_table_text(text: str, prefix: str) -> str:
    """Add deterministic invisible claim anchors to each displayed table row."""
    if any(ANCHOR_RE.match(line) for line in text.splitlines()):
        return text
    output: list[str] = []
    header_count = 0
    row_count = 0
    before_midrule = True
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == r"\midrule":
            before_midrule = False
        if stripped.endswith(r"\\") and not stripped.startswith((r"\toprule", r"\midrule", r"\bottomrule")):
            if before_midrule:
                header_count += 1
                claim_id = f"A3-{prefix}-H{header_count:02d}"
            else:
                row_count += 1
                claim_id = f"A3-{prefix}-R{row_count:02d}"
            output.append(f"% SRC {claim_id}")
        output.append(line)
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def anchor_generated_tables(package_root: Path) -> list[Path]:
    changed: list[Path] = []
    generated = package_root / "generated"
    for filename, prefix in TABLE_PREFIXES.items():
        path = generated / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated table: {path}")
        original = path.read_text(encoding="utf-8")
        anchored = anchor_generated_table_text(original, prefix)
        if anchored != original:
            path.write_text(anchored, encoding="utf-8")
            changed.append(path)
    return changed
