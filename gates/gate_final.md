# Gate: final artefacts on Drive (brief section 5, steps 3-4) — PASS

Checked 06:19 UTC Sat 10 Oct 2026 by the report session. Command for every directory item:
`rclone check <local> <drive> --one-way --size-only` (eval tree with the uploader's excludes). Log: logs/final_checks.log.
Copies were made with `rclone copy` / `rclone copyto` only (no sync, delete, purge or move). No Hugging Face access.
Single files were compared by byte size (`stat` against `rclone lsl`), because rclone check needs directories.

| Item (local -> Drive) | Result |
|---|---|
| final_models/molmoact2_sim_v1_023400/pretrained_model -> models/molmoact2_sim_v1_023400/pretrained_model | 7 matching, 0 differences |
| final_models/act_single_bs32_030000/pretrained_model -> models/act_single_bs32_030000/pretrained_model | 7 matching, 0 differences |
| final_models/smolvla_single_bs64_020000/pretrained_model -> models/smolvla_single_bs64_020000/pretrained_model | 7 matching, 0 differences |
| final_models/act_wide_bs32_030000/pretrained_model -> models/act_wide_bs32_030000/pretrained_model | 7 matching, 0 differences |
| final_models/act_multi_nolang_bs32_030000/pretrained_model -> models/act_multi_nolang_bs32_030000/pretrained_model | 7 matching, 0 differences |
| final_models/smolvla_multi_bs64_015000/pretrained_model -> models/smolvla_multi_bs64_015000/pretrained_model | 7 matching, 0 differences |
| datasets/*.tar.zst (3 tars) -> datasets/ | 3 matching, 0 differences |
| eval/ (all runs; excludes rollouts, dryrun, test_bplus) -> eval/ | 512 matching, 0 differences |
| bundles/*.bundle (10 git bundles) -> code/ | 10 matching, 0 differences |
| gates/ (20 files, before this file) -> gates/ | 20 matching, 0 differences |
| deliverables/ -> deliverables/ | 14 matching, 0 differences |
| train/p23_*.json, p23max_*.json (14) -> p23/ | 14 matching, 0 differences |
| SCOREBOARD.md -> SCOREBOARD.md | byte size matches (51739) |
| LEDGER.md -> ledger/LEDGER.md | byte size matches at 06:10 copy; re-copied at the end |
| replication/LAPTOP.md -> replication/LAPTOP.md | byte size matches (5806) |
| outputs_large/FINAL_ARTEFACTS.txt -> FINAL_ARTEFACTS.txt | byte size matches (4606) |

Code: the 10 bundles match by size. Their git commits are listed in outputs_large/FINAL_ARTEFACTS.txt. The report branch report/overnight is pushed to origin.

Not uploaded, by design (FINAL_ARTEFACTS.txt, last section): intermediate checkpoints, uncompressed datasets, eval rollouts, the dropped B+ test tree, the B+ round outputs. The JEPA outputs were backed up to the HF dataset VietHwang/so101-jepa-judge on 9 Oct, before the HF stop.

Spot-check by main session 06:24 UTC: MD5 (not size-only) match on the 6 models, 3 dataset tars, 10 bundles, gates, SCOREBOARD.md, FINAL_ARTEFACTS.txt.

Verdict: every item in FINAL_ARTEFACTS.txt is on Drive with a passing size check.
Caveat: size-only checks do not compare content. A spot-check by content is recommended before the owner relies on any single file.
