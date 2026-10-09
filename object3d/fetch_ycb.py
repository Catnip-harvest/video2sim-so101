"""Download the three YCB ground-truth scans used to validate the mesh -> physics path.

    python -I object3d/fetch_ycb.py            # ~24 MB of archives, ~59 MB unpacked

Source: the official YCB Object and Model Set bucket (Calli et al., 2015),
http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/ ("Google scanner", 16k-polygon
textured meshes). Licence of the data: Creative Commons Attribution 4.0 International
(CC BY 4.0), stated on that page (checked 9 Oct 2026). Attribution is in
object3d/third_party/README.md. The files land in object3d/third_party/ycb/, which is
gitignored; nothing downloaded is committed.

Masses and sizes are the official ones from the YCB object list
(http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://ycb-benchmarks.s3.amazonaws.com/data/google/{id}_google_16k.tgz"
DEST = Path(__file__).resolve().parent / "third_party" / "ycb"
MIN_FREE_BYTES = 1_000_000_000  # stop if the disk has less than 1 GB free

# id: (sha256 of the archive as downloaded on 9 Oct 2026, official mass kg, official size)
OBJECTS = {
  "077_rubiks_cube": ("f5a44bde4605f402247e7f4b5e2ef93469dabfd07062cb2b67efdcdbe088587c",
                      0.094, "57 x 57 x 57 mm"),
  "005_tomato_soup_can": ("264f28559475e8e548b20420a7b6db944b4b27471039a73733d2c08a4ba2cc3d",
                          0.349, "66 mm dia x 101 mm"),
  "025_mug": ("4b640afc1baaa940b66948d1e59426d8da2c5a16558c7c7a118bbe28d0737de5",
              0.118, "80 mm dia x 82 mm"),
}


def main() -> None:
  free = shutil.disk_usage(DEST.parent if DEST.parent.exists() else Path.cwd()).free
  if free < MIN_FREE_BYTES:
    sys.exit(f"only {free / 1e9:.2f} GB free; need 1 GB. Stopping.")
  arch = DEST / "_archives"
  arch.mkdir(parents=True, exist_ok=True)
  for oid, (sha, _, _) in OBJECTS.items():
    tgz = arch / f"{oid}_google_16k.tgz"
    if not tgz.is_file():
      url = BASE.format(id=oid)
      print("downloading", url)
      urllib.request.urlretrieve(url, tgz)
    got = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if got != sha:
      sys.exit(f"{tgz.name}: sha256 {got} != expected {sha}")
    with tarfile.open(tgz) as tf:
      tf.extractall(DEST, filter="data")  # refuses absolute paths / links outside DEST
    print("ok", oid, "->", DEST / oid / "google_16k")


if __name__ == "__main__":
  main()
