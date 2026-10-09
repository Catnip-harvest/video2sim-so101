PASS: MolmoAct2-SO100_101 20-step bf16 smoke, loss finite; bs16 1.23 s/step, 146 GB GPU peak; 2 epochs of 300 sim eps ~6.8 h
# Gate 1b: MolmoAct2 smoke (FINAL_DESIGN §10), 9 Oct ~09:56-09:59 UTC
- bs8 (render job sharing CPU): update 0.73 s/step, data 1.16 s/step, 81 GB lerobot / 90.8 GB GPU peak, loss 3.56 -> 2.90.
- bs16 (CPU free, 8 loader workers): update 1.23 s, data 0.09 s, 131 GB / 146 GB peak of 192, loss 3.78 -> 2.42.
- Estimate at 13 samples/s: 2 epochs of 300 sim eps (~158k frames) ~6.8 h; + friend's 334 real eps (~209k frames) ~15.7 h.
- Logs: logs/smoke_molmoact2_bs8.log, logs/smoke_molmoact2_bs16.log. Script: video2sim-so101/env/smoke_molmoact2.sh.
