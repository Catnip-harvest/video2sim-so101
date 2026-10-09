"""OPTIONAL hook, NOT run on this laptop: per-frame task-progress curve for the
human demo video from a frozen video-language reward model (Robometer-4B via
LeRobot). Intended for the MI300X box; the 4B Qwen3-VL backbone needs ~9 GB of
GPU memory, more than the RTX 4060's 8 GB.

Setup on the GPU box (from a LeRobot source checkout):
    pip install -e ".[robometer]"

Run:
    python -I tools/progress_curve_robometer.py VIDEO.mp4 \
        --task "pick up the red block and put it in the bowl" \
        --spec outputs/<run>/task_spec.json --device cuda

LeRobot's compute_reward() returns only the LAST frame's progress, so the curve
is built from growing prefixes of the video: progress(t) = reward of frames
[0, t], with up to --max-frames frames subsampled from each prefix. The curve is
written to <spec dir>/progress_curve.json and linked from task_spec.json
("progress_curve"). API per https://huggingface.co/docs/lerobot/en/robometer
(checked 2026-10-09); this script itself is unverified until run on the box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_frames(path: str, max_side: int = 448) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, times = [], []
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s = min(1.0, max_side / max(f.shape[:2]))
        if s < 1:
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        times.append(k / fps)
        k += 1
    return np.stack(frames).astype(np.uint8), np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--task", required=True, help="natural-language task description")
    ap.add_argument("--spec", help="task_spec.json to link the curve into")
    ap.add_argument("--points", type=int, default=24, help="number of prefix cut points on the curve")
    ap.add_argument("--max-frames", type=int, default=8, help="frames per prefix given to the model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="lerobot/Robometer-4B")
    args = ap.parse_args()

    from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
    from lerobot.rewards.robometer.modeling_robometer import ROBOMETER_FEATURE_PREFIX
    from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

    cfg = RobometerConfig(pretrained_path=args.model, device=args.device, reward_output="progress")
    model = RobometerRewardModel.from_pretrained(cfg.pretrained_path, config=cfg)
    encoder = RobometerEncoderProcessorStep(base_model_id=cfg.base_model_id, use_multi_image=cfg.use_multi_image,
                                            use_per_frame_progress_token=cfg.use_per_frame_progress_token,
                                            max_frames=args.max_frames)
    frames, times = read_frames(args.video)
    cuts = np.unique(np.linspace(max(1, args.max_frames // 2), len(frames), args.points).astype(int))
    curve = []
    for end in cuts:
        idx = np.unique(np.linspace(0, end - 1, min(args.max_frames, end)).astype(int))
        encoded = encoder.encode_samples([(frames[idx], args.task)])
        batch = {f"{ROBOMETER_FEATURE_PREFIX}{key}": value for key, value in encoded.items()}
        progress = float(model.compute_reward(batch).reshape(-1)[0])
        curve.append({"t": round(float(times[end - 1]), 4), "progress": round(progress, 4)})
        print(curve[-1], flush=True)

    out = Path(args.spec).parent if args.spec else Path(".")
    payload = {"model": args.model, "task": args.task, "method": "prefix last-frame progress", "curve": curve}
    (out / "progress_curve.json").write_text(json.dumps(payload, indent=2))
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        spec["progress_curve"] = "progress_curve.json"
        Path(args.spec).write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
