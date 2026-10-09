#!/usr/bin/env python
"""Run a Python entry point under an emulated laptop-GPU memory cap (P2.3).

Plain words: we tell PyTorch to refuse GPU memory above a cap, then run the
normal training or eval command in the same process and report the peak.

Wording to use everywhere (verbatim from the overnight brief, P2.3):

    "fits under an emulated 5.0 GiB PyTorch allocator cap (proxy for a 6 GB
    laptop card; context and driver overhead on CUDA not measured)"

Caveat: `torch.cuda.set_per_process_memory_fraction` caps only PyTorch's
caching allocator, not the HIP/CUDA context, MIOpen/rocBLAS workspaces or a
laptop's display memory, and it emulates memory, not speed. So the caps are set
to the *usable* budget: 5.0 GiB for the "6 GB-class" row, 7.0 GiB for the
"8 GB-class" row. Timing measured here is MI300X timing (not laptop).

Usage (inside container `sprint`, cwd = repo root):

    # training: console script looked up by name, argv forwarded unchanged
    python env/capped.py --cap-gb 5.0 -- lerobot-train --policy.type=act ...

    # any python file (sys.path[0] = the script's directory, like `python x.py`)
    python env/capped.py --cap-gb 7.0 -- eval/sim_eval.py --workers 1 ...

    # a module
    python env/capped.py --cap-gb 5.0 -- -m some.module --flag

    # env-var form (policy_io-style use): CAP_GB is read when --cap-gb is absent
    CAP_GB=5.0 python env/capped.py -- eval/sim_eval.py --workers 1 ...

    # self-test (GPU, < 2 GB, seconds): 0.5 GiB cap, 0.3 GiB fits, 1.0 GiB OOMs
    python env/capped.py --self-test

How sim_eval runs under the cap: wrap it as above and keep `--workers 1`
(the default, and the only mode the eval queue allows). With `--workers > 1`
sim_eval starts *spawned* pool processes, which do NOT inherit the cap.
Code that loads a policy in-process can also call `apply_cap_from_env()` from
this module before its first CUDA allocation (a no-op when CAP_GB is unset).

At exit one line is printed to stdout (and to --result-json if given):

    CAPPED_RESULT {"cap_gb": 5.0, "total_gb": ..., "max_memory_allocated_gb": ...,
                   "max_memory_reserved_gb": ..., "exit_status": 0, "oom": false, ...}

Exit status: the target's own status, or 3 when it died of an out-of-memory
error (torch.OutOfMemoryError, or a RuntimeError/other error whose message says
"out of memory").
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import runpy
import shutil
import sys
import time
import traceback
from pathlib import Path

OOM_EXIT = 3
GIB = 2**30


def apply_cap(cap_gb: float, device: int = 0) -> dict:
  """Set the PyTorch allocator cap to cap_gb GiB on `device`. Returns info."""
  import torch

  if not torch.cuda.is_available():
    raise RuntimeError("capped.py: no CUDA/HIP device visible")
  total = torch.cuda.get_device_properties(device).total_memory
  frac = cap_gb * GIB / total
  if not 0.0 < frac <= 1.0:
    raise ValueError(f"cap {cap_gb} GiB is outside (0, total={total / GIB:.1f} GiB]")
  torch.cuda.set_per_process_memory_fraction(frac, device)
  torch.cuda.reset_peak_memory_stats(device)
  return {"cap_gb": cap_gb, "total_gb": round(total / GIB, 3), "fraction": frac, "device": device}


def apply_cap_from_env(device: int = 0) -> dict | None:
  """Apply the cap from env var CAP_GB if it is set; otherwise do nothing."""
  v = os.environ.get("CAP_GB", "").strip()
  if not v:
    return None
  return apply_cap(float(v), device)


def is_oom(exc: BaseException) -> bool:
  """True for torch.OutOfMemoryError or any error whose text says 'out of memory'."""
  try:
    import torch

    oom_t = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_t is not None and isinstance(exc, oom_t):
      return True
  except Exception:  # noqa: BLE001
    pass
  e: BaseException | None = exc
  while e is not None:  # also look through chained causes
    if "out of memory" in str(e).lower():
      return True
    e = e.__cause__ or e.__context__
  return False


def peak_stats(device: int = 0) -> dict:
  import torch

  return {
    "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(device) / GIB, 4),
    "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved(device) / GIB, 4),
    "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
    "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
  }


def _exit_code(code) -> int:
  if code is None:
    return 0
  if isinstance(code, int):
    return code
  print(code, file=sys.stderr)  # sys.exit("message") semantics
  return 1


def run_target(target: list[str]) -> int:
  """Run target in-process with sys.argv forwarded. Returns its exit status."""
  if not target:
    raise SystemExit("capped.py: nothing to run after `--`")
  if target[0] in ("python", "python3", sys.executable):
    target = target[1:]
  if target[0] == "-m":
    mod, rest = target[1], target[2:]
    sys.argv = [mod] + rest
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
    return 0

  name, rest = target[0], target[1:]
  path = Path(name)
  if name.endswith(".py") or path.is_file():
    path = path.resolve()
    sys.argv = [str(path)] + rest
    sys.path.insert(0, str(path.parent))
    runpy.run_path(str(path), run_name="__main__")
    return 0

  eps = [ep for ep in md.entry_points(group="console_scripts") if ep.name == name]
  if eps:
    fn = eps[0].load()
    sys.argv = [name] + rest
    return _exit_code(fn())

  exe = shutil.which(name)
  if exe:  # a python wrapper script on PATH
    sys.argv = [exe] + rest
    runpy.run_path(exe, run_name="__main__")
    return 0
  raise SystemExit(f"capped.py: cannot resolve target {name!r} (not a file, module or console script)")


def emit(result: dict, result_json: str | None) -> None:
  line = "CAPPED_RESULT " + json.dumps(result, sort_keys=True)
  sys.stdout.flush()
  print(line, flush=True)
  if result_json:
    Path(result_json).parent.mkdir(parents=True, exist_ok=True)
    Path(result_json).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


def self_test(device: int = 0) -> int:
  """0.5 GiB cap: a 0.3 GiB tensor must fit, a 1.0 GiB tensor must OOM."""
  import torch

  info = apply_cap(0.5, device)
  res = dict(info)
  x = torch.empty(int(0.3 * GIB), dtype=torch.uint8, device=f"cuda:{device}")
  res["alloc_0p3_ok"] = True
  res["after_0p3"] = peak_stats(device)
  del x
  torch.cuda.empty_cache()
  try:
    y = torch.empty(int(1.0 * GIB), dtype=torch.uint8, device=f"cuda:{device}")
    del y
    res["alloc_1p0_oom"] = False
  except Exception as e:  # noqa: BLE001
    res["alloc_1p0_oom"] = is_oom(e)
    res["alloc_1p0_error"] = type(e).__name__
  res.update(peak_stats(device))
  res["pass"] = bool(res["alloc_0p3_ok"] and res["alloc_1p0_oom"])
  print("CAPPED_SELFTEST " + json.dumps(res, sort_keys=True), flush=True)
  return 0 if res["pass"] else 1


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  if "--" in argv:
    i = argv.index("--")
    own, target = argv[:i], argv[i + 1 :]
  else:
    own, target = argv, []
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--cap-gb", type=float, default=None, help="allocator cap in GiB (default: env CAP_GB)")
  ap.add_argument("--device", type=int, default=0)
  ap.add_argument("--result-json", default=None, help="also write the result JSON here")
  ap.add_argument("--self-test", action="store_true")
  a = ap.parse_args(own)

  if a.self_test:
    return self_test(a.device)
  cap = a.cap_gb if a.cap_gb is not None else (float(os.environ["CAP_GB"]) if os.environ.get("CAP_GB") else None)
  if cap is None:
    ap.error("give --cap-gb or set CAP_GB")

  # The cap is set here, before the target (and lerobot) is even imported.
  info = apply_cap(cap, a.device)
  os.environ["CAP_GB"] = str(cap)  # visible to the target, e.g. for its own logging
  result = dict(info, target=target, oom=False, error=None)
  t0 = time.time()
  status = 1
  try:
    status = run_target(target)
  except SystemExit as e:
    status = _exit_code(e.code)
  except BaseException as e:  # noqa: BLE001
    traceback.print_exc()
    if is_oom(e):
      result["oom"] = True
      status = OOM_EXIT
    else:
      status = 1
    result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
  finally:
    result["wall_s"] = round(time.time() - t0, 1)
    try:
      result.update(peak_stats(a.device))
    except Exception as e:  # noqa: BLE001
      result["peak_error"] = repr(e)
    result["exit_status"] = status
    emit(result, a.result_json)
  return status


if __name__ == "__main__":
  sys.exit(main())
