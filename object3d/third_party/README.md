# Third-party data used by object3d

Nothing in this folder except this file is committed. `object3d/fetch_ycb.py` downloads it.

## YCB Object and Model Set (ground-truth scans)

| Object | File | Official mass and size |
|---|---|---|
| 077_rubiks_cube | `https://ycb-benchmarks.s3.amazonaws.com/data/google/077_rubiks_cube_google_16k.tgz` (5.8 MB) | 94 g, 57 × 57 × 57 mm |
| 005_tomato_soup_can | `https://ycb-benchmarks.s3.amazonaws.com/data/google/005_tomato_soup_can_google_16k.tgz` (10.6 MB) | 349 g, 66 mm dia × 101 mm |
| 025_mug | `https://ycb-benchmarks.s3.amazonaws.com/data/google/025_mug_google_16k.tgz` (7.7 MB) | 118 g, 80 mm dia × 82 mm |

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the
  official download page http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/
  (checked 9 Oct 2026). The code on that page is MIT.
- **Attribution:** B. Calli, A. Walsman, A. Singh, S. Srinivasa, P. Abbeel and A. M. Dollar,
  "Benchmarking in Manipulation Research: Using the Yale-CMU-Berkeley Object and Model Set,"
  IEEE Robotics & Automation Magazine 22(3), 2015. Meshes: Google scanner, 16k polygons.
- **Masses and sizes:** official object list,
  http://www.ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf
- **Checksums (sha256 of the archives):** pinned in `object3d/fetch_ycb.py`.

Anything made from these scans (the MJCF folders in `object3d/out/`) is a CC BY 4.0
derivative; keep the attribution above with it if it is ever shared.
