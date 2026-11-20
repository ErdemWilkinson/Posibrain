"""Cok dronlu baz cizgiler. Kullanim: python baselines_multi.py <hucre> <dron_sayisi> [bolum]
Klasik tarama: her dron kendi dikey bandinda (A/D genislik) boustrophedon yapar."""
import sys
import time
import numpy as np
import envm as E
from baselines import wilson


class RandomM:
    name = "rastgele"

    def __init__(self, e, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, e):
        a = self.rng.integers(0, 9, e.nA)
        a[np.repeat(e.has, e.D).astype(bool)] = 11
        return a


class LawnMowerM:
    name = "klasik tarama (bantli)"

    def __init__(self, e, seed=0):
        D = e.D
        self.row = np.zeros(e.nA, np.int64)
        d = np.tile(np.arange(D), e.n)
        self.lo = d * e.A / D
        self.hi = (d + 1) * e.A / D

    def act(self, e):
        x, y = e.x.reshape(-1), e.y.reshape(-1)
        alive = e.bat.reshape(-1) > 0
        east = (self.row % 2) == 0
        ty = self.row * E.STEP_M
        edge = np.where(east, x > self.hi - 1.5 * E.STEP_M, x < self.lo + 1.5 * E.STEP_M)
        low, high = y < ty - 20, y > ty + 20
        a = np.where(east, np.where(low, 1, np.where(high, 7, 0)),
                     np.where(low, 3, np.where(high, 5, 4)))
        a = np.where(edge, 2, a)
        self.row = self.row + (edge & alive).astype(np.int64)
        a[np.repeat(e.has, e.D).astype(bool)] = 11
        return a


class LawnMowerHighM(LawnMowerM):
    name = "tarama yuksek irtifa"

    def act(self, e):
        a = super().act(e)
        up = (e.alt.reshape(-1) < 2) & (np.repeat(e.has, e.D) == 0)
        return np.where(up, 9, a)


def evaluate(cls, cells, D, n_ep, seed0=1_000_000):
    e = E.SearchEnv(n_ep, cells=cells, drones=D)
    e.reset(np.ones(n_ep, bool), seeds=np.arange(seed0, seed0 + n_ep))
    pol = cls(e)
    info = None
    for _ in range(E.MAX_STEPS + 5):
        if e.done.all():
            break
        _, _, _, info = e.step(pol.act(e))
    s = info["succ"].astype(bool)
    k = int(s.sum())
    lo, hi = wilson(k, n_ep)
    return dict(ad=cls.name, k=k, rate=k / n_ep, lo=lo, hi=hi,
                steps=float(info["steps"][s].mean()) if k else float("nan"),
                wrong=float(info["wrong"].mean()), hazard=float(info["hazard"].mean()),
                veto=float(info["veto"].mean()), cover=float(info["cover"].mean()))


if __name__ == "__main__":
    cells = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000
    print(f"harita {cells * E.CELL / 1000:.1f} km, {D} dron, {n_ep} bolum (tohum {seed0}+), pil {E.BATTERY_S:.0f} s")
    for cls in (RandomM, LawnMowerM, LawnMowerHighM):
        t = time.time()
        r = evaluate(cls, cells, D, n_ep, seed0)
        print(f"{r['ad']:22s} bulma %{100*r['rate']:.1f} (%95 GA {100*r['lo']:.1f}-{100*r['hi']:.1f}) "
              f"adim {r['steps']:.0f} yanlis {r['wrong']:.2f} tehlike {r['hazard']:.1f} "
              f"veto {r['veto']:.1f} kapsama %{100*r['cover']:.0f}  [{time.time()-t:.1f} s]")
