from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rule_policy_ablation_covers_all_rows_and_variants(tmp_path: Path) -> None:
    output = tmp_path / "ablation"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_rule_policy_ablation.py"),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["variants"]["full"]["metrics"]["rows"] == 509
    assert len(report["feature_prevalence"]) == 4
    assert len(report["variants"]) == 9
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 509
