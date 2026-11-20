"""Cok dronlu arama simulasyonu (envm). env.py'nin D dronlu hali.

Kararlar (TASARIM.md): tum dronlar ayni politikayi kullanir (parametre paylasimi). Keşfedilmiş maske ve aday
tespit tum filo icin ortaktir (haberlesme varsayimi: kusursuz paylasim, ilk surum). Kardeslere carpmama
OGRENILMEZ, sert bir kural: hedef nokta baska bir dronun SEP metre icindeyse hareket reddedilir (dron havada
bekler). Ogrenilen kisim is bolumu. Rapor edilen aday dogruysa tum filo odul alir.
Hepsi olculmemis varsayim; gercek donanima aktarim iddiasi yoktur.
"""
import math
import numpy as np
from numba import njit, prange

CELL = 10.0
STEP_M = 90.0
SPEED = 10.0
BATTERY_S = 1400.0
RADIUS = (20.0, 45.0, 90.0)
P_LOW, P_MID, P_HIGH = 0.97, 0.93, 0.70
CPEN_LOW, CPEN_MID, CPEN_HIGH = 0.15, 0.35, 0.70
SWAMP_T = 0.74
SCAN_BONUS = 0.10
SCAN_S = 10.0
ALT_S = 5.0
REPORT_S = 2.0
SWAMP_ENERGY = 2.0
N_BIRDS = 2
BIRD_STEP = 60.0
BIRD_FALSE_P = 0.25
BIRD_ATTACK_R = 60.0
BIRD_ATTACK_COST = 15.0
CAND_NOISE = 25.0
REPORT_OK_R = 50.0
MAX_WIND = 1.5
MAX_STEPS = 400
SEP = 40.0                     # m, minimum dron arasi mesafe (sert kural)
VETO_S = 3.0                   # s, reddedilen hareketin maliyeti
P = 15
PB = 4
G = 20
N_ACT = 12
NVEC = 15
MASK64 = (1 << 64) - 1

import os as _os
for _k in list(globals()):
    _ev = _os.environ.get("SIM_" + _k)
    if _ev is not None and isinstance(globals()[_k], (int, float)) and not isinstance(globals()[_k], bool):
        globals()[_k] = type(globals()[_k])(float(_ev))
P_DET = (P_LOW, P_MID, P_HIGH)
CANOPY_PEN = (CPEN_LOW, CPEN_MID, CPEN_HIGH)


def _seed_state(seed):
    z = (int(seed) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return np.uint64(z if z else 1)


@njit(inline='always')
def _rnd(st, i):
    x = st[i]
    x ^= (x << np.uint64(13))
    x ^= (x >> np.uint64(7))
    x ^= (x << np.uint64(17))
    st[i] = x
    return float(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit
def _noise(out, st, i, cells):
    H = out.shape[0]
    gh = cells + 2
    g = np.empty((gh, gh))
    for a in range(gh):
        for b in range(gh):
            g[a, b] = _rnd(st, i)
    for y in range(H):
        fy = y * cells / H
        y0 = int(fy)
        ty = fy - y0
        ty = ty * ty * (3.0 - 2.0 * ty)
        for x in range(H):
            fx = x * cells / H
            x0 = int(fx)
            tx = fx - x0
            tx = tx * tx * (3.0 - 2.0 * tx)
            a = g[y0, x0] * (1.0 - tx) + g[y0, x0 + 1] * tx
            b = g[y0 + 1, x0] * (1.0 - tx) + g[y0 + 1, x0 + 1] * tx
            out[y, x] = a * (1.0 - ty) + b * ty


@njit(inline='always')
def _segdist(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = min(1.0, max(0.0, t))
    ex = ax + t * dx - px
    ey = ay + t * dy - py
    return math.sqrt(ex * ex + ey * ey)


@njit(inline='always')
def _stamp(ex, xm, ym, r):
    H = ex.shape[0]
    rc = r / CELL
    cx = int(xm / CELL)
    cy = int(ym / CELL)
    ri = int(rc) + 1
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            if dx * dx + dy * dy <= rc * rc:
                xx = cx + dx
                yy = cy + dy
                if 0 <= xx < H and 0 <= yy < H:
                    ex[yy, xx] = 1


@njit
def _reset_one(i, st, canopy, swamp, explored, x, y, alt, bat, steps, px, py, bx, by,
               wind, has, cx, cy, done, succ, wrong, hazard, veto, c1, c2, c3):
    H = canopy.shape[1]
    A = H * CELL
    D = x.shape[1]
    n1 = np.empty((H, H))
    n2 = np.empty((H, H))
    n3 = np.empty((H, H))
    _noise(n1, st, i, c1)
    _noise(n2, st, i, c2)
    _noise(n3, st, i, c3)
    for yy in range(H):
        for xx in range(H):
            v = 0.6 * n1[yy, xx] + 0.4 * n2[yy, xx]
            canopy[i, yy, xx] = min(1.0, max(0.0, (v - 0.42) * 3.0))
            swamp[i, yy, xx] = 1 if n3[yy, xx] > SWAMP_T else 0
            explored[i, yy, xx] = 0
    want_canopy = _rnd(st, i) < 0.7
    pxx = 0.0
    pyy = 0.0
    for t in range(80):
        gx = int(_rnd(st, i) * H)
        gy = int(_rnd(st, i) * H)
        pxx = (gx + 0.5) * CELL
        pyy = (gy + 0.5) * CELL
        if (not want_canopy) or canopy[i, gy, gx] > 0.5:
            break
    px[i] = pxx
    py[i] = pyy
    for b in range(bx.shape[1]):
        bx[i, b] = _rnd(st, i) * A
        by[i, b] = _rnd(st, i) * A
    ang = _rnd(st, i) * 2.0 * math.pi
    spd = _rnd(st, i) * MAX_WIND
    wind[i, 0] = math.cos(ang) * spd
    wind[i, 1] = math.sin(ang) * spd
    for d in range(D):
        x[i, d] = (d + 0.1) * A / D
        y[i, d] = 0.0
        alt[i, d] = 1
        bat[i, d] = BATTERY_S
    steps[i] = 0
    has[i] = 0
    cx[i] = 0.0
    cy[i] = 0.0
    done[i] = 0
    succ[i] = 0
    wrong[i] = 0
    hazard[i] = 0
    veto[i] = 0


@njit(parallel=True)
def _reset_masked(mask, st, canopy, swamp, explored, x, y, alt, bat, steps, px, py, bx, by,
                  wind, has, cx, cy, done, succ, wrong, hazard, veto, c1, c2, c3):
    for i in prange(mask.shape[0]):
        if mask[i]:
            _reset_one(i, st, canopy, swamp, explored, x, y, alt, bat, steps, px, py, bx, by,
                       wind, has, cx, cy, done, succ, wrong, hazard, veto, c1, c2, c3)


@njit(parallel=True)
def _step(act, st, canopy, swamp, explored, x, y, alt, bat, steps, px, py, bx, by,
          wind, has, cx, cy, done, succ, wrong, hazard, veto, rew):
    n = act.shape[0]
    D = act.shape[1]
    H = canopy.shape[1]
    A = H * CELL
    isc = 1.0 / D   # bireysel maliyetler dron sayisina bolunur; takim odulleri (basari, yanlis rapor) bolunmez
    for i in prange(n):
        for d in range(D):
            rew[i, d] = 0.0
        if done[i]:
            continue
        for b in range(bx.shape[1]):
            bx[i, b] = min(max(bx[i, b] + (_rnd(st, i) - 0.5) * 2.0 * BIRD_STEP, 0.0), A)
            by[i, b] = min(max(by[i, b] + (_rnd(st, i) - 0.5) * 2.0 * BIRD_STEP, 0.0), A)
        for d in range(D):
            if done[i]:
                break
            if bat[i, d] <= 0.0:
                continue
            a = act[i, d]
            ax = x[i, d]
            ay = y[i, d]
            nx = ax
            ny = ay
            dt = 0.0
            moved = False
            scan = False
            report = False
            if a < 8:
                ang = a * (math.pi / 4.0)
                nx = ax + math.cos(ang) * STEP_M + wind[i, 0] * STEP_M / SPEED
                ny = ay + math.sin(ang) * STEP_M + wind[i, 1] * STEP_M / SPEED
                nx = min(max(nx, 0.0), A - 1e-3)
                ny = min(max(ny, 0.0), A - 1e-3)
                vetoed = False
                for e in range(D):
                    if e != d and bat[i, e] > 0.0:
                        if (nx - x[i, e]) ** 2 + (ny - y[i, e]) ** 2 < SEP * SEP:
                            vetoed = True
                            break
                if vetoed:
                    nx = ax
                    ny = ay
                    dt = VETO_S
                    veto[i] += 1
                    rew[i, d] -= 0.05 * isc
                else:
                    moved = True
                    dt = STEP_M / SPEED
                    if swamp[i, int(ny / CELL), int(nx / CELL)] != 0:
                        dt *= SWAMP_ENERGY
                        hazard[i] += 1
            elif a == 8:
                scan = True
                dt = SCAN_S
            elif a == 9:
                if alt[i, d] < 2:
                    alt[i, d] += 1
                dt = ALT_S
            elif a == 10:
                if alt[i, d] > 0:
                    alt[i, d] -= 1
                dt = ALT_S
            else:
                report = True
                dt = REPORT_S
            r = RADIUS[alt[i, d]]
            if moved or scan:
                ns = int(STEP_M / CELL) + 1 if moved else 1
                for k in range(ns + 1):
                    t = k / ns
                    _stamp(explored[i], ax + t * (nx - ax), ay + t * (ny - ay), r)
                dd0 = _segdist(px[i], py[i], ax, ay, nx, ny)
                if dd0 <= r:
                    cp = canopy[i, int(py[i] / CELL), int(px[i] / CELL)]
                    p = P_DET[alt[i, d]] * (1.0 - CANOPY_PEN[alt[i, d]] * cp)
                    if scan:
                        p = min(0.99, p + SCAN_BONUS)
                    if _rnd(st, i) < p:
                        has[i] = 1
                        ang2 = _rnd(st, i) * 2.0 * math.pi
                        rr = _rnd(st, i) * CAND_NOISE
                        cx[i] = px[i] + math.cos(ang2) * rr
                        cy[i] = py[i] + math.sin(ang2) * rr
                for b in range(bx.shape[1]):
                    db = _segdist(bx[i, b], by[i, b], ax, ay, nx, ny)
                    if db <= r and _rnd(st, i) < BIRD_FALSE_P:
                        has[i] = 1
                        cx[i] = bx[i, b]
                        cy[i] = by[i, b]
            for b in range(bx.shape[1]):
                if math.sqrt((bx[i, b] - nx) ** 2 + (by[i, b] - ny) ** 2) < BIRD_ATTACK_R:
                    bat[i, d] -= BIRD_ATTACK_COST
                    hazard[i] += 1
                    rew[i, d] -= 0.2 * isc
                    break
            was_alive = bat[i, d] > 0.0
            x[i, d] = nx
            y[i, d] = ny
            bat[i, d] -= dt
            rew[i, d] -= 0.002 * dt * isc
            if was_alive and bat[i, d] <= 0.0:
                rew[i, d] -= 1.0 * isc
            if report:
                if has[i]:
                    dd = math.sqrt((cx[i] - px[i]) ** 2 + (cy[i] - py[i]) ** 2)
                    if dd <= REPORT_OK_R:
                        succ[i] = 1
                        done[i] = 1
                        bonus = 10.0 + 5.0 * max(0.0, bat[i, d]) / BATTERY_S
                        for e in range(D):
                            rew[i, e] += bonus
                    else:
                        wrong[i] += 1
                        has[i] = 0
                        for e in range(D):
                            rew[i, e] -= 3.0
                else:
                    rew[i, d] -= 0.5 * isc
        steps[i] += 1
        if not done[i]:
            alive = 0
            for d in range(D):
                if bat[i, d] > 0.0:
                    alive += 1
            if alive == 0 or steps[i] >= MAX_STEPS:
                done[i] = 1


@njit(parallel=True)
def _obs(loc, glo, vec, canopy, swamp, explored, x, y, alt, bat, steps, wind, has, cx, cy):
    n = canopy.shape[0]
    D = x.shape[1]
    H = canopy.shape[1]
    A = H * CELL
    pool = H // G
    for i in prange(n):
        for gy in range(G):
            for gx in range(G):
                s = 0.0
                for dy in range(pool):
                    for dx in range(pool):
                        s += explored[i, gy * pool + dy, gx * pool + dx]
                gv = s / (pool * pool)
                for d in range(D):
                    glo[i * D + d, gy, gx] = gv
        for d in range(D):
            k = i * D + d
            ci = int(x[i, d] / CELL)
            cj = int(y[i, d] / CELL)
            for a in range(P):
                for b in range(P):
                    y0 = cj + (a - P // 2) * PB - PB // 2
                    x0 = ci + (b - P // 2) * PB - PB // 2
                    se = 0.0
                    sc = 0.0
                    ss = 0.0
                    si = 0.0
                    for dy in range(PB):
                        for dx in range(PB):
                            yy = y0 + dy
                            xx = x0 + dx
                            if 0 <= yy < H and 0 <= xx < H:
                                si += 1.0
                                e = float(explored[i, yy, xx])
                                se += e
                                sc += e * canopy[i, yy, xx]
                                ss += e * swamp[i, yy, xx]
                    f = 1.0 / (PB * PB)
                    loc[k, 0, a, b] = se * f
                    loc[k, 1, a, b] = sc * f
                    loc[k, 2, a, b] = ss * f
                    loc[k, 3, a, b] = si * f
            vec[k, 0] = max(0.0, bat[i, d]) / BATTERY_S
            vec[k, 1] = 1.0 if alt[i, d] == 0 else 0.0
            vec[k, 2] = 1.0 if alt[i, d] == 1 else 0.0
            vec[k, 3] = 1.0 if alt[i, d] == 2 else 0.0
            vec[k, 4] = x[i, d] / A
            vec[k, 5] = y[i, d] / A
            vec[k, 6] = steps[i] / MAX_STEPS
            vec[k, 7] = float(has[i])
            if has[i]:
                vec[k, 8] = min(1.0, max(-1.0, (cx[i] - x[i, d]) / 500.0))
                vec[k, 9] = min(1.0, max(-1.0, (cy[i] - y[i, d]) / 500.0))
            else:
                vec[k, 8] = 0.0
                vec[k, 9] = 0.0
            vec[k, 10] = wind[i, 0] / MAX_WIND
            vec[k, 11] = wind[i, 1] / MAX_WIND
            best = 1e18
            bdx = 0.0
            bdy = 0.0
            found = 0.0
            for e in range(D):
                if e != d and bat[i, e] > 0.0:
                    dx2 = x[i, e] - x[i, d]
                    dy2 = y[i, e] - y[i, d]
                    dist2 = dx2 * dx2 + dy2 * dy2
                    if dist2 < best:
                        best = dist2
                        bdx = dx2
                        bdy = dy2
                        found = 1.0
            vec[k, 12] = min(1.0, max(-1.0, bdx / 500.0))
            vec[k, 13] = min(1.0, max(-1.0, bdy / 500.0))
            vec[k, 14] = found


class SearchEnv:
    def __init__(self, n, cells=140, seed0=0, drones=2):
        assert cells % G == 0
        self.n, self.D, self.H = n, drones, cells
        self.nA = n * drones
        self.A = cells * CELL
        sc = max(1, int(round(self.A / 1000.0)))
        self.c1, self.c2, self.c3 = 5 * sc, 14 * sc, 6 * sc
        z = lambda: np.zeros(n)
        zd = lambda: np.zeros((n, drones))
        self.canopy = np.zeros((n, cells, cells), np.float32)
        self.swamp = np.zeros((n, cells, cells), np.uint8)
        self.explored = np.zeros((n, cells, cells), np.uint8)
        self.x, self.y, self.bat = zd(), zd(), zd()
        self.alt = np.zeros((n, drones), np.int64)
        self.px, self.py, self.cx, self.cy = z(), z(), z(), z()
        self.steps = np.zeros(n, np.int64)
        self.bx = np.zeros((n, N_BIRDS))
        self.by = np.zeros((n, N_BIRDS))
        self.wind = np.zeros((n, 2))
        self.has = np.zeros(n, np.uint8)
        self.done = np.zeros(n, np.uint8)
        self.succ = np.zeros(n, np.uint8)
        self.wrong = np.zeros(n, np.int64)
        self.hazard = np.zeros(n, np.int64)
        self.veto = np.zeros(n, np.int64)
        self.st = np.ones(n, np.uint64)
        self.seed = np.zeros(n, np.int64)
        self.counter = seed0
        self.rew = zd()
        self.loc = np.zeros((self.nA, 4, P, P), np.float32)
        self.glo = np.zeros((self.nA, G, G), np.float32)
        self.vec = np.zeros((self.nA, NVEC), np.float32)
        self.reset(np.ones(n, bool))

    def reset(self, mask, seeds=None):
        mask = np.asarray(mask, bool)
        for i in np.nonzero(mask)[0]:
            s = seeds[i] if seeds is not None else self.counter
            if seeds is None:
                self.counter += 1
            self.seed[i] = s
            self.st[i] = _seed_state(s)
        _reset_masked(mask.astype(np.uint8), self.st, self.canopy, self.swamp, self.explored,
                      self.x, self.y, self.alt, self.bat, self.steps, self.px, self.py,
                      self.bx, self.by, self.wind, self.has, self.cx, self.cy, self.done,
                      self.succ, self.wrong, self.hazard, self.veto, self.c1, self.c2, self.c3)
        return self.obs()

    def obs(self):
        _obs(self.loc, self.glo, self.vec, self.canopy, self.swamp, self.explored, self.x,
             self.y, self.alt, self.bat, self.steps, self.wind, self.has, self.cx, self.cy)
        return self.loc, self.glo, self.vec

    def step(self, act, autoreset=False):
        act = np.ascontiguousarray(np.asarray(act).reshape(self.n, self.D), np.int64)
        _step(act, self.st, self.canopy, self.swamp, self.explored, self.x, self.y, self.alt,
              self.bat, self.steps, self.px, self.py, self.bx, self.by, self.wind, self.has,
              self.cx, self.cy, self.done, self.succ, self.wrong, self.hazard, self.veto, self.rew)
        info = dict(done=self.done.copy(), succ=self.succ.copy(), wrong=self.wrong.copy(),
                    hazard=self.hazard.copy(), veto=self.veto.copy(), steps=self.steps.copy(),
                    bat=self.bat.mean(1), cover=self.explored.reshape(self.n, -1).mean(1))
        rew = self.rew.reshape(-1).copy()
        done = np.repeat(self.done, self.D)
        if autoreset and self.done.any():
            self.reset(self.done.astype(bool))
        return self.obs(), rew, done, info
