"""Cok dron politikasi tani: greedy ve olasiliksal rollout'ta eylem dagilimi, hareket, veto, sonuc."""
import os
import sys
import importlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
from net import build

w, D, cells = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
E = importlib.import_module("envm")
m = build(E, 1.0, False)
m.load_weights(w)
n = 100
names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "tara", "alt+", "alt-", "rapor"]


def prep(e):
    return (np.ascontiguousarray(e.loc.transpose(0, 2, 3, 1)), e.glo[..., None].copy(), e.vec.copy())


for mode in ("greedy", "olasiliksal"):
    e = E.SearchEnv(n, cells=cells, drones=D)
    e.reset(np.ones(n, bool), seeds=np.arange(3_000_000, 3_000_000 + n))
    hist = np.zeros(12)
    x0 = e.x.copy()
    for t in range(E.MAX_STEPS):
        if e.done.all():
            break
        lg = m(list(prep(e)), training=False)[0].numpy()
        if mode == "greedy":
            a = lg.argmax(1)
        else:
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            a = np.array([np.random.choice(12, p=r) for r in p])
        act = a.reshape(n, D)
        alive = (e.bat > 0) & (e.done[:, None] == 0)
        for k in range(12):
            hist[k] += ((act == k) & alive).sum()
        e.step(a)
    tot = hist.sum()
    print(f"--- {mode}: basari %{100*e.succ.mean():.0f} adim ort {e.steps.mean():.0f} veto ort {e.veto.mean():.1f} "
          f"yanlis ort {e.wrong.mean():.2f} kapsama %{100*e.explored.reshape(n,-1).mean():.0f} "
          f"ort. yer degistirme {np.abs(e.x - x0).mean():.0f} m")
    print("   eylem dagilimi: " + ", ".join(f"{names[k]} %{100*hist[k]/tot:.0f}" for k in range(12) if hist[k] > 0))
