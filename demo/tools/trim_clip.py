"""Cut a time window out of a clip (re-encoded with OpenCV, mp4v).

    python -I tools/trim_clip.py IN.mp4 OUT.mp4 --start 0.0 --end 3.0
"""
import argparse

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, required=True)
a = ap.parse_args()
cap = cv2.VideoCapture(a.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
k = n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if a.start <= k / fps < a.end:
        out.write(f)
        n += 1
    k += 1
out.release()
print(f"wrote {a.dst}: {n} frames, {w}x{h} @ {fps:.2f} fps")
