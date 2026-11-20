"""Elle yapilandirilmis budama: tam genislikteki agirliklardan (w=1) w genislikli kucuk ag uretir.
Her katmanda cikis kanallarini/birimlerini L1 onemine gore secer, sonraki katmanin girislerini buna gore keser.
Kullanim: python prune.py <tam.h5> <genislik> <cikis.h5> [--drones D] [--aux]
Sonra: python ppo.py --width <genislik> --init <cikis.h5> ... (ince ayar) ve --width <genislik> (sifirdan kontrol)."""
import os
import sys
import importlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from net import build

src, width, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
D = int(sys.argv[sys.argv.index("--drones") + 1]) if "--drones" in sys.argv else 1
aux = "--aux" in sys.argv
E = importlib.import_module("envm" if D > 1 else "env")

full = build(E, 1.0, aux)
full.load_weights(src)
small = build(E, width, aux)
W = {l.name: l.get_weights() for l in full.layers if l.get_weights()}
S = {l.name: l for l in small.layers if l.get_weights()}


def top(score, k):
    return np.sort(np.argsort(-score)[:k])


def kdim(name):
    return S[name].get_weights()[0].shape[-1]


new = {}


def conv_branch(c1, c2, dn, hw_pos, in_idx=None):
    k1, b1 = W[c1]
    idx1 = top(np.abs(k1).sum((0, 1, 2)), kdim(c1))
    new[c1] = [k1[..., idx1], b1[idx1]]
    k2, b2 = W[c2]
    k2 = k2[:, :, idx1, :]
    idx2 = top(np.abs(k2).sum((0, 1, 2)), kdim(c2))
    new[c2] = [k2[..., idx2], b2[idx2]]
    kd, bd = W[dn]
    cfull = k2.shape[-1]
    kd = kd.reshape(hw_pos, cfull, kd.shape[-1])[:, idx2, :].reshape(hw_pos * len(idx2), -1)
    idxd = top(np.abs(kd).sum(0), kdim(dn))
    new[dn] = [kd[:, idxd], bd[idxd]]
    return idxd


iL = conv_branch("l_c1", "l_c2", "l_d", 16)
iG = conv_branch("g_c1", "g_c2", "g_d", 25)
kv, bv = W["v_d"]
iV = top(np.abs(kv).sum(0), kdim("v_d"))
new["v_d"] = [kv[:, iV], bv[iV]]
k1, b1 = W["h1"]
nL, nG = W["l_d"][0].shape[-1], W["g_d"][0].shape[-1]
rows = np.concatenate([iL, nL + iG, nL + nG + iV])
k1 = k1[rows, :]
i1 = top(np.abs(k1).sum(0), kdim("h1"))
new["h1"] = [k1[:, i1], b1[i1]]
k2, b2 = W["h2"]
k2 = k2[i1, :]
i2 = top(np.abs(k2).sum(0), kdim("h2"))
new["h2"] = [k2[:, i2], b2[i2]]
for head in ("logits", "value", "p_find"):
    if head in W:
        kh, bh = W[head]
        new[head] = [kh[i2, :], bh]
for n, ws in new.items():
    S[n].set_weights(ws)
small.save_weights(dst)
print(f"budandi: w={width}, parametre {full.count_params()} -> {small.count_params()} "
      f"(~{small.count_params()/1024:.0f} KB int8)  -> {dst}")
