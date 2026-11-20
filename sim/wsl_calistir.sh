#!/bin/bash
# Kullanim (Windows'tan): wsl -d Ubuntu-22.04 -u root -- bash <bu dosyanin /mnt/c yolu> <script.py> [argumanlar]
# Kod Windows klasorunde yazilir, burada ASCII bir yola kopyalanip /opt/tfgpu ortaminda calistirilir.
set -e
SRC=${DRON_SIM_WIN:?DRON_SIM_WIN ortam degiskenini .py dosyalarinin bulundugu Windows klasorune ayarla, ornek: export DRON_SIM_WIN=/mnt/c/Users/<sen>/dron_sim}
mkdir -p /root/dron_sim
cp "$SRC"/*.py /root/dron_sim/
cd /root/dron_sim
/opt/tfgpu/bin/python "$@"
