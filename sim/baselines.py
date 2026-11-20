"""Baz cizgiler ve degerlendirme. Kullanim: python baselines.py [hucre_sayisi] [bolum_sayisi]"""
import sys
import time
import numpy as np
import env as E


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


class RandomPolicy:
    name = "rastgele"

    def __init__(self, n, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, e):
        a = self.rng.integers(0, 9, e.n)
        a[e.has.astype(bool)] = 11
        return a


class LawnMower:
    """Boustrophedon: satirlar arasi STEP_M, ruzgar icin y duzeltmeli. Aday varsa rapor eder."""
    name = "klasik tarama"

    def __init__(self, n, seed=0):
        self.row = np.zeros(n, np.int64)

    def act(self, e):
        n, A = e.n, e.A
        a = np.zeros(n, np.int64)
        east = (self.row % 2) == 0
        ty = self.row * E.STEP_M
        edge = np.where(east, e.x > A - 1.5 * E.STEP_M, e.x < 1.5 * E.STEP_M)
        low = e.y < ty - 20
        high = e.y > ty + 20
        a = np.where(east, np.where(low, 1, np.where(high, 7, 0)),
                     np.where(low, 3, np.where(high, 5, 4)))
        a = np.where(edge, 2, a)
        self.row = self.row + (edge & (e.done == 0)).astype(np.int64)
        a[e.has.astype(bool)] = 11
        return a


class LawnMowerHigh(LawnMower):
    """Ince ayarli kural baz cizgisi: once yuksek irtifaya cik (genis serit, her nokta ~2 kez kaplanir),
    sonra ayni boustrophedon. Ogrenmenin kurala karsi gercek degerini olcmek icin."""
    name = "tarama yuksek irtifa"

    def act(self, e):
        a = super().act(e)
        up = (e.alt < 2) & (e.has == 0)
        return np.where(up, 9, a)


def evaluate(policy_cls, cells, n_ep, seed0=1_000_000):
    e = E.SearchEnv(n_ep, cells=cells)
    e.reset(np.ones(n_ep, bool), seeds=np.arange(seed0, seed0 + n_ep))
    pol = policy_cls(n_ep)
    for _ in range(E.MAX_STEPS + 5):
        if e.done.all():
            break
        _, _, _, info = e.step(pol.act(e))
    s = info["succ"].astype(bool)
    k = int(s.sum())
    lo, hi = wilson(k, n_ep)
    return dict(ad=policy_cls.name, bulma=k / n_ep, lo=lo, hi=hi,
                sure=float(info["steps"][s].mean()) if k else float("nan"),
                pil=float(info["bat"][s].mean() / E.BATTERY_S) if k else float("nan"),
                yanlis=float(info["wrong"].mean()), tehlike=float(info["hazard"].mean()),
                kapsama=float(info["cover"].mean()))


if __name__ == "__main__":
    cells = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
    print(f"harita {cells * E.CELL / 1000:.1f} km, {n_ep} bolum (tohum {seed0}+), pil {E.BATTERY_S:.0f} s")
    for cls in (RandomPolicy, LawnMower, LawnMowerHigh):
        t = time.time()
        r = evaluate(cls, cells, n_ep, seed0)
        print(f"{r['ad']:14s} bulma %{100*r['bulma']:.1f} (%95 GA {100*r['lo']:.1f}-{100*r['hi']:.1f}) "
              f"adim {r['sure']:.0f} pil {r['pil']:.2f} yanlis {r['yanlis']:.2f} "
              f"tehlike {r['tehlike']:.1f} kapsama %{100*r['kapsama']:.0f}  [{time.time()-t:.1f} s]")
