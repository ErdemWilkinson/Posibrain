#!/bin/bash
# Sirali gece kuyrugu: bir seferde TEK is (RAM). WSL icinde /root/dron_sim'den calisir.
# 1) basari-olasiligi head'i  2) budama w=0.5 (ince ayar ve sifirdan kontrol)  3) 2 dron  4) 4 dron  5) test kumesi baz cizgileri
cd /root/dron_sim
PY=/opt/tfgpu/bin/python
LOG=runs/gece.log
mkdir -p runs
step() { echo "[$(date +%H:%M)] $*" | tee -a $LOG; }
run() { "$@" 2>&1 | grep -v -E "cuda|cuDNN|cuFFT|cuBLAS|XLA|absl|factory|I0000|WARNING|INFO|fully_quantize|f32" | tee -a $LOG; }
WIN=${DRON_SIM_WIN:?DRON_SIM_WIN ortam degiskenini .py dosyalarinin bulundugu Windows klasorune ayarla, ornek: export DRON_SIM_WIN=/mnt/c/Users/<sen>/dron_sim}

step "1/5 elle head: basari olasiligi (1 km)"
run $PY ppo.py --iters 800 --eval_every 20 --base 0.806 --base_hi 0.823 --aux 1 --name t1b_aux
run $PY quant.py --weights runs/t1b_aux/best.weights.h5 --out runs/t1b_aux --aux 1 --n 2000

step "2/5 elle budama w=0.5"
run $PY prune.py runs/t1b_1km/best.weights.h5 0.5 runs/p50_init.h5
run $PY ppo.py --width 0.5 --init runs/p50_init.h5 --iters 300 --eval_every 20 --base 0.806 --base_hi 0.823 --lr 1e-4 --name t1b_p50_ft
run $PY quant.py --weights runs/t1b_p50_ft/best.weights.h5 --out runs/t1b_p50_ft --width 0.5 --n 2000
run $PY ppo.py --width 0.5 --iters 800 --eval_every 20 --base 0.806 --base_hi 0.823 --name t1b_p50_scratch
run $PY quant.py --weights runs/t1b_p50_scratch/best.weights.h5 --out runs/t1b_p50_scratch --width 0.5 --n 2000

step "3/5 2 dron, 1.4 km"
run $PY ppo.py --drones 2 --cells 140 --envs 128 --iters 800 --eval_every 20 --base 0.820 --base_hi 0.851 --name t2_2dron
run $PY quant.py --weights runs/t2_2dron/best.weights.h5 --out runs/t2_2dron --drones 2 --cells 140 --n 1000

step "4/5 4 dron, 2 km"
run $PY ppo.py --drones 4 --cells 200 --envs 64 --iters 800 --eval_every 20 --base 0.814 --base_hi 0.846 --name t3_4dron
run $PY quant.py --weights runs/t3_4dron/best.weights.h5 --out runs/t3_4dron --drones 4 --cells 200 --n 1000

step "5/5 test kumesi baz cizgileri (tohum 2.000.000+)"
run $PY baselines_multi.py 140 2 1000 2000000
run $PY baselines_multi.py 200 4 1000 2000000

# sonuclari Windows klasorune kopyala
mkdir -p "$WIN/runs"
cp -r runs/* "$WIN/runs/" 2>/dev/null
step "GECE KUYRUGU BITTI"
