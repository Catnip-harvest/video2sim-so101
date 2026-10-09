"""Download the three model checkpoints the extractor needs, each into its own
directory under manip/demo/third_party/.

    python -I tools/fetch_models.py

Every model here is Apache-2.0 and needs no account or licence sign-up:

  * MediaPipe HandLandmarker (float16) - Google, Apache-2.0 (model card:
    "Model Card Hand Tracking (Lite/Full) with Fairness Oct 2021").
  * The same MediaPipe palm-detector + hand-landmark models converted to ONNX
    by the OpenCV Zoo (opencv/palm_detection_mediapipe,
    opencv/handpose_estimation_mediapipe) - Apache-2.0. Used when the
    mediapipe package's native DLL cannot load (blocked by Windows
    Application Control on this laptop).
  * SAM 2.1 hiera-tiny, Hugging Face transformers format - Meta, Apache-2.0.
  * Depth Anything V2 Small, transformers format - Apache-2.0. (The Base and
    Large checkpoints are CC-BY-NC-4.0 and are deliberately not used.)

Only safetensors/ONNX weights + JSON configs are fetched (no pickled .pt files).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
THIRD_PARTY = DEMO_DIR / "third_party"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

HF_MODELS = {
    # local dir name : (repo id, files to fetch)
    "sam2.1_hiera_tiny": (
        "facebook/sam2.1-hiera-tiny",
        ["config.json", "model.safetensors", "preprocessor_config.json",
         "processor_config.json", "video_preprocessor_config.json"],
    ),
    "opencv_zoo_palmdet": (
        "opencv/palm_detection_mediapipe",
        ["LICENSE", "README.md", "palm_detection_mediapipe_2023feb.onnx"],
    ),
    "opencv_zoo_handpose": (
        "opencv/handpose_estimation_mediapipe",
        ["LICENSE", "README.md", "handpose_estimation_mediapipe_2023feb.onnx"],
    ),
    "depth_anything_v2_small": (
        "depth-anything/Depth-Anything-V2-Small-hf",
        ["config.json", "model.safetensors", "preprocessor_config.json"],
    ),
}


def fetch_hand_landmarker() -> None:
    target_dir = THIRD_PARTY / "mediapipe_hand_landmarker"
    target = target_dir / "hand_landmarker.task"
    if target.exists():
        print(f"ok   {target}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"get  {HAND_LANDMARKER_URL}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, target)
    print(f"ok   {target} ({target.stat().st_size / 1e6:.1f} MB)")


def fetch_hf_models() -> None:
    from huggingface_hub import hf_hub_download  # Python API; the `hf` CLI is blocked here

    for local_name, (repo_id, files) in HF_MODELS.items():
        target_dir = THIRD_PARTY / local_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if (target_dir / filename).exists():
                continue
            print(f"get  {repo_id}/{filename}")
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=target_dir)
        print(f"ok   {target_dir}")


if __name__ == "__main__":
    fetch_hand_landmarker()
    fetch_hf_models()
    sys.exit(0)
