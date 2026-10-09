PASS: 10 Hz + fast render + variety: 0.92 attempts/s (16 workers), 96.8 % accepted (pass marks 0.5 eps/s, 70 %)
# Gate 1a: render throughput (FINAL_DESIGN §7), re-measured 9 Oct ~10:15 UTC
- Final config: --render-quality fast --record-hz 10 --variety --keep-rejects, 16 workers, 2 cams, 256 px.
  60/62 accepted (2 arm_table_collision), 0.92 attempts/s; 10 Hz dataset loads (18/18 LeRobotDataset checks).
  Log: logs/bench_v10_n60.log (EXIT=0).
- Earlier, 30 Hz fast, 15 workers: 0.34-0.36 eps/s (borderline). Image throughput, 19 workers:
  full 45 img/s, shadow 117, fast 421.
- 10 Hz for every source, per owner: mixing 30 Hz and 10 Hz would make action chunks span different times.
