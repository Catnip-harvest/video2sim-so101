# Stage 7 gate: realism number — NOT DONE

October's realism score (branch `jepa/judge`, `jepa/realism.py`) is a Frechet distance between frozen
V-JEPA 2 ViT-L clip features of sim and real cam_head clips (200 vs 200, 20 splits): sim vs real 1329, real vs real
floor 15. Recomputing it needs V-JEPA 2 ViT-L features for ~400 new clips. October extracted them on the GPU;
today the GPU is off limits (two training jobs; a third GPU process livelocks the MI300X) and the CPU budget (8
processes) is taken by the 1000-episode render until it finishes. A CPU run of ViT-L over 400 x 16-frame clips was
not attempted.

To run it later on the twin's renders (after `gen_friend_bowl_n1000_render` finishes):
`git show origin/jepa/judge:jepa/extract.py` / `jepa/realism.py`, pointing the sim side at
`/root/work/outputs_large/friendrig/gen_friend_bowl_n1000_render/lerobot` (cam_head videos).
