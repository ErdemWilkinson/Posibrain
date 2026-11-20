# Posibrain

**A tiny edge-AI "search brain" for a rescue drone, learned from scratch in a fast simulation, then quantized to int8 for an ESP32-class chip.**

Posibrain by Erdem Aslan. Code and weights: Apache 2.0 (see `LICENSE`, `NOTICE`).

> **Read this first.** Everything here is a **simulation**. Every number in the simulator (battery, speed, detection probabilities, swamp, birds) is an unmeasured assumption. Nothing was run on a drone or on a microcontroller. The int8 sizes are real file sizes, the success rates are simulation results. Do not read them as field performance.

## Özet (Türkçe)

Dron, insanın yerini bilmeden (ipucusuz) bir haritada arama yapmayı **sıfır ağırlıktan** öğreniyor. Ortam hızlı, görselsiz bir simülasyon (prosedürel orman, bataklık, yırtıcı kuş, rüzgar, pil). Ağ 52 bin parametre, int8'de **60,8 KB**. 1 km'de tek dronda öğrenen ajan, en iyi elle yazılmış kuralı (yüksek irtifada tara) küçük bir farkla geçti. Yarı genişlikte, sıfırdan eğitilmiş bir ağ (13,7 bin parametre, **21,8 KB**) kuralla başa baş. Çoklu dron, başarı olasılığı head'i ve budama denemeleri **başarısız ya da kötü** çıktı. Hepsi aşağıda açık açık yazıyor.

## What the agent does

- Map: a procedurally generated forest / swamp world, 1 km x 1 km for one drone (10 m cells). Every episode has a new terrain and a new random person location.
- No hint: the drone does not know where the person is. It starts at the home corner with a full battery.
- Actions (12): move in 8 directions, scan (hover), altitude up, altitude down, **report**.
- The reward comes mainly from a correct **report** (position within 50 m of the person). False alarms (birds produce fake detections) are penalized. A small energy cost applies per second.
- Altitude is a real trade-off: high altitude gives a wide swath but forest canopy hides people much more; low altitude is accurate but narrow.
- Policy: small CNN (local map patch + global explored mask + scalar vector), PPO, trained on GPU (TensorFlow, WSL Ubuntu). The simulator runs on CPU with numba.

## Results (1 km, 1 drone)

Held-out test: seeds 2,000,000+, 2,000 episodes, 95% Wilson confidence intervals. Checkpoints were chosen on different seeds (1,000,000+). One training run per configuration, so run-to-run variance is **not** measured (evaluation during training swings by tens of points between checkpoints).

| Policy | Success rate (95% CI) | int8 size |
|---|---|---|
| Random flight | 27.4% (25.5 - 29.4) | - |
| Lawn-mower sweep, mid altitude (rule) | 74.9% (72.9 - 76.7) | - |
| Lawn-mower sweep, high altitude (best rule I could write) | 80.6% (78.8 - 82.3) | - |
| **Posibrain, float** (52,420 params) | **84.2% (82.5 - 85.7)** | - |
| **Posibrain, int8** | **83.2% (81.5 - 84.8)** | **60.8 KB** |
| Posibrain nano width 0.5, trained from scratch, float (13,656 params) | 82.0% (80.2 - 83.6) | - |
| Posibrain nano width 0.5, int8 | 80.7% (78.9 - 82.4) | **21.8 KB** |

What this says, and what it does not:

- The float policy beats the best rule by about 3.6 points, and the confidence intervals only just separate (82.5 vs 82.3). A fair summary is "slightly better than a tuned rule in this simulator".
- int8 loses about 1 point, which is within the noise.
- The half-width network is on par with the rule at under 22 KB.
- An earlier version of this benchmark compared against a weak rule (mid-altitude sweep, 74.4%). A simple "fly high" rule scored 93.5% in the first simulator calibration, which showed the simulator had no real altitude trade-off. I re-calibrated (altitude-dependent canopy penalty) and the numbers above are from the corrected simulator. The first calibration's run is kept in `results/runs/t1_1km` and is **superseded**.

## What did not work (kept on purpose)

1. **Multi-drone (2 and 4 drones): failed.** All drones share one policy; they share the explored map and the candidate detection; a hard rule (not learned) keeps drones 40 m apart. First attempt: the policy collapsed, and the 4-drone policy learned to hover and toggle altitude. Cause: individual costs were multiplied by the number of drones while the team success reward was not. After scaling individual costs by 1/D: 2 drones reach 58.1% (55.0 - 61.1) on the test set against 80.1% for the rule baseline, and 4 drones still score 0%. Not solved. (`results/runs/t2_2dron`, `t3_4dron` are the first attempt, `t2b_2dron`, `t3b_4dron` the second.)
2. **Success-probability head ("hand-added head"): failed.** An extra head predicting P(success) was trained alongside the policy. v1 used labels that were biased toward short episodes. v2 uses a TD(lambda) target, but it is still badly calibrated (Brier 0.285 against 0.179 for a constant predictor) and the policy got worse (77.2% float). So this repo does **not** give calibrated probabilities yet.
3. **Pruning did not beat training from scratch.** Structured pruning to width 0.5 drops the policy to about 24% before fine-tuning. After 300 fine-tuning iterations it reaches 74.1% (int8 73.0%), while the same architecture trained from scratch for 800 iterations reaches 82.0%. The budgets are not equal (300 vs 800 iterations), so this is a weak negative, not proof that pruning is worse.

## Not done

- No "return home" action (in the design, not built).
- No radar, sound or thermal sensors. The "sensor" is an abstract detection probability model.
- No real drone, no real mud physics, no on-device latency or RAM measurement.
- Only one training seed per configuration.

## Layout

```
sim/env.py            single-drone simulator (numba, deterministic per seed)
sim/envm.py           multi-drone simulator
sim/net.py            policy network (width multiplier, optional P(success) head)
sim/ppo.py            PPO training (GPU)  --drones --width --aux --init
sim/baselines*.py     random + lawn-mower rule baselines, Wilson intervals
sim/prune.py          structured pruning of the policy
sim/quant.py          TFLite int8 export + float vs int8 comparison in the simulator
sim/diag_multi.py     diagnosis of collapsed multi-drone policies
sim/gece*.sh          the overnight experiment queues that produced the results
docs/TASARIM.md       design notes (Turkish)
results/              training logs, quantization results, calibration
weights/              final policies (.tflite int8 and float .weights.h5)
```

The helper `sim/wsl_calistir.sh` runs the code from WSL Ubuntu with a TensorFlow-GPU environment (`/opt/tfgpu`). Simulator constants can be overridden with environment variables (`SIM_BATTERY_S=1500`); every setting needs a fresh process because numba freezes constants at first compile.

## Licence

Apache License 2.0 for code and weights. The simulation uses no third-party data. Attribution: keep `NOTICE` and credit "Posibrain by Erdem Aslan".
