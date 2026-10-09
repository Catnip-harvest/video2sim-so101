"""Tests for env/capped.py (GPU, tiny: < 1.1 GiB, a few seconds each).

  python -m pytest -q -p no:cacheprovider small/tests/test_capped.py

Skipped when no GPU is visible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAPPED = ROOT / "env" / "capped.py"

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _run(args, tmp_path=None):
  p = subprocess.run([sys.executable, str(CAPPED), *args], capture_output=True, text=True, timeout=180)
  return p


def _parse(stdout: str, tag: str) -> dict:
  lines = [ln for ln in stdout.splitlines() if ln.startswith(tag + " ")]
  assert lines, stdout
  return json.loads(lines[-1][len(tag) + 1 :])


def test_self_test_0p3_fits_1p0_ooms():
  p = _run(["--self-test"])
  r = _parse(p.stdout, "CAPPED_SELFTEST")
  assert p.returncode == 0, p.stdout + p.stderr
  assert r["alloc_0p3_ok"] and r["alloc_1p0_oom"]
  assert r["max_memory_reserved_gb"] <= 0.5 + 1e-6


def test_wrapper_forwards_argv_and_reports_oom(tmp_path):
  script = tmp_path / "target.py"
  script.write_text(
    "import sys, torch\n"
    "gb = float(sys.argv[1])\n"
    "x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device='cuda')\n"
    "print('ALLOCATED', gb)\n"
  )
  ok = _run(["--cap-gb", "0.5", "--", str(script), "0.2"])
  r = _parse(ok.stdout, "CAPPED_RESULT")
  assert ok.returncode == 0 and "ALLOCATED 0.2" in ok.stdout
  assert r["oom"] is False and r["exit_status"] == 0 and r["cap_gb"] == 0.5
  assert 0.19 < r["max_memory_allocated_gb"] <= 0.5

  bad = _run(["--cap-gb", "0.5", "--", str(script), "1.0"])
  r = _parse(bad.stdout, "CAPPED_RESULT")
  assert bad.returncode == 3 and r["oom"] is True and r["exit_status"] == 3


def test_env_var_cap(tmp_path):
  script = tmp_path / "t.py"
  script.write_text("import os; print('CAP', os.environ['CAP_GB'])\n")
  import os

  p = subprocess.run([sys.executable, str(CAPPED), "--", str(script)], capture_output=True, text=True,
                     timeout=180, env=dict(os.environ, CAP_GB="0.75"))
  r = _parse(p.stdout, "CAPPED_RESULT")
  assert p.returncode == 0 and r["cap_gb"] == 0.75 and "CAP 0.75" in p.stdout
