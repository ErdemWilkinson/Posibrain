#!/bin/bash
# Ikinci kuyruk (duzeltilmis cok-dron odulu + yanlilik giderilmis basari-olasiligi head'i). Sirali, tek is.
cd /root/dron_sim
PY=/opt/tfgpu/bin/python
LOG=runs/gece2.log
mkdir -p runs
step() { echo "[$(date +%H:%M)] $*" | tee -a $LOG; }
run() { "$@" 2>&1 | grep -v -E "cuda|cuDNN|cuFFT|cuBLAS|XLA|absl|factory|I0000|WARNING|INFO|fully_quantize|f32" | tee -a $LOG; }
WIN=${DRON_SIM_WIN:?DRON_SIM_WIN ortam degiskenini .py dosyalarinin bulundugu Windows klasorune ayarla, ornek: export DRON_SIM_WIN=/mnt/c/Users/<sen>/dron_sim}

step "1/3 2 dron, 1.4 km (duzeltilmis odul)"
run $PY ppo.py --drones 2 --cells 140 --envs 128 --iters 800 --eval_every 20 --base 0.820 --base_hi 0.851 --name t2b_2dron
run $PY quant.py --weights runs/t2b_2dron/best.weights.h5 --out runs/t2b_2dron --drones 2 --cells 140 --n 1000
cp -r runs/* "$WIN/runs/" 2>/dev/null

step "2/3 4 dron, 2 km (duzeltilmis odul)"
run $PY ppo.py --drones 4 --cells 200 --envs 64 --iters 800 --eval_every 20 --base 0.814 --base_hi 0.846 --name t3b_4dron
run $PY quant.py --weights runs/t3b_4dron/best.weights.h5 --out runs/t3b_4dron --drones 4 --cells 200 --n 1000
cp -r runs/* "$WIN/runs/" 2>/dev/null

step "3/3 elle head v2: basari olasiligi (TD hedefi), 1 km"
run $PY ppo.py --iters 800 --eval_every 20 --base 0.806 --base_hi 0.823 --aux 1 --name t1c_aux
run $PY quant.py --weights runs/t1c_aux/best.weights.h5 --out runs/t1c_aux --aux 1 --n 2000
cp -r runs/* "$WIN/runs/" 2>/dev/null
step "IKINCI KUYRUK BITTI"
