#!/usr/bin/env python3
"""Run before Git push: verify source alignment and current annotation export."""
import subprocess
import sys
from pathlib import Path

script=Path(__file__).resolve().with_name('export_dataset_502_labels.py')
cmd=[sys.executable,str(script),'--check-only',*sys.argv[1:]]
raise SystemExit(subprocess.call(cmd))
