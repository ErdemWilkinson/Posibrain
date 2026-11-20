"""PPO (TensorFlow, GPU) ile sifirdan arama politikasi. Ortam numba ile CPU'da, ag GPU'da.
Kullanim: python ppo.py --cells 100 --iters 600 --name t1_1km
Degerlendirme: egitimde hic gorulmemis tohumlar (1_000_000+), %95 Wilson GA, baz cizgiyle ayni tohumlar.
"""
import os
import sys
import time
import glob
import shutil
import argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import importlib
import numpy as np
import tensorflow as tf
from baselines import wilson

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=int, default=100)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--mb", type=int, default=4096)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--ent", type=float, default=0.01)
ap.add_argument("--gamma", type=float, default=0.995)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--eval_n", type=int, default=1000)
ap.add_argument("--base", type=float, default=0.744, help="klasik tarama bulma orani (baz cizgi)")
ap.add_argument("--base_hi", type=float, default=0.770)
ap.add_argument("--name", default="run")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--drones", type=int, default=1, help=">1 ise envm (cok dron, parametre paylasimli)")
ap.add_argument("--init", default="", help="baslangic agirliklari (.h5), ornegin onceki kademe")
ap.add_argument("--width", type=float, default=1.0, help="ag genislik carpani (budama karsilastirmasi)")
ap.add_argument("--aux", type=int, default=0, help="1: basari-olasiligi head'i ekle (elle head)")
ap.add_argument("--aux_w", type=float, default=0.2)
args = ap.parse_args()
D = max(1, args.drones)
E = importlib.import_module("envm" if D > 1 else "env")

tf.random.set_seed(args.seed)
np.random.seed(args.seed)
OUT = os.path.join("/root/dron_sim/runs", args.name)
os.makedirs(OUT, exist_ok=True)
WIN = os.environ.get("DRON_SIM_WIN")  # istege bagli: sonuclari Windows klasorune geri kopyalar


def sync_back():
    if WIN is None:
        return
    d = os.path.join(WIN, "runs", args.name)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*")):
        shutil.copy(f, d)


from net import build

model = build(E, args.width, args.aux)
if args.init:
    model.load_weights(args.init)
    print("agirliklar yuklendi:", args.init, flush=True)
opt = tf.keras.optimizers.Adam(args.lr)
print(f"parametre: {model.count_params()}  (~{model.count_params()/1024:.0f} KB int8)", flush=True)


def prep(e):
    return (np.ascontiguousarray(e.loc.transpose(0, 2, 3, 1)), e.glo[..., None].copy(), e.vec.copy())


@tf.function
def act_fn(loc, glo, vec):
    o = model([loc, glo, vec], training=False)
    logits, value = o[0], o[1]
    a = tf.random.categorical(logits, 1)[:, 0]
    lp = tf.gather(tf.nn.log_softmax(logits), a, batch_dims=1)
    pf = tf.sigmoid(o[2][:, 0]) if args.aux else tf.zeros_like(value[:, 0])
    return a, lp, value[:, 0], pf


@tf.function
def greedy_fn(loc, glo, vec):
    o = model([loc, glo, vec], training=False)
    return tf.argmax(o[0], axis=1)


@tf.function
def value_fn(loc, glo, vec):
    return model([loc, glo, vec], training=False)[1][:, 0]


@tf.function
def pfind_fn(loc, glo, vec):
    return tf.sigmoid(model([loc, glo, vec], training=False)[2][:, 0])


@tf.function
def train_step(loc, glo, vec, act, old_lp, adv, ret, lab, msk):
    with tf.GradientTape() as tape:
        o = model([loc, glo, vec], training=True)
        logits, value = o[0], o[1]
        logp_all = tf.nn.log_softmax(logits)
        lp = tf.gather(logp_all, act, batch_dims=1)
        ratio = tf.exp(lp - old_lp)
        pl = -tf.reduce_mean(tf.minimum(ratio * adv, tf.clip_by_value(ratio, 0.8, 1.2) * adv))
        vl = 0.5 * tf.reduce_mean(tf.square(ret - value[:, 0]))
        ent = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, -1))
        loss = pl + 0.5 * vl - args.ent * ent
        if args.aux:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=lab, logits=o[2][:, 0])
            loss += args.aux_w * tf.reduce_sum(msk * bce) / (tf.reduce_sum(msk) + 1e-6)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 0.5)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return pl, vl, ent


def mk(n, s0=0):
    if D > 1:
        return E.SearchEnv(n, cells=args.cells, seed0=s0, drones=D)
    return E.SearchEnv(n, cells=args.cells, seed0=s0)


env = mk(args.envs)
eval_env = mk(args.eval_n)


def evaluate(calib=False):
    n = args.eval_n
    eval_env.reset(np.ones(n, bool), seeds=np.arange(1_000_000, 1_000_000 + n))
    info = None
    P_log, A_log, T_log = [], [], []
    for t in range(E.MAX_STEPS + 5):
        if eval_env.done.all():
            break
        o = prep(eval_env)
        if calib and args.aux:
            p = pfind_fn(*o).numpy().reshape(n, D).mean(1)
            P_log.append(p)
            A_log.append(eval_env.done.copy() == 0)
            T_log.append(t)
        a = greedy_fn(*o).numpy()
        _, _, _, info = eval_env.step(a)
    s = info["succ"].astype(bool)
    cal = None
    if calib and args.aux and P_log:
        Pm, Am = np.array(P_log), np.array(A_log)
        y = np.broadcast_to(s.astype(np.float64), Pm.shape)
        p, yy = Pm[Am], y[Am]
        base = float(yy.mean())   # adim agirlikli taban oran (sabit tahmin icin adil karsilastirma)
        brier = float(np.mean((p - yy) ** 2))
        brier_base = float(np.mean((base - yy) ** 2))
        bins = np.linspace(0, 1, 11)
        rel = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mk_ = (p >= b0) & (p < b1 + (1e-9 if b1 == 1 else 0))
            if mk_.sum() >= 20:
                rel.append((float(p[mk_].mean()), float(yy[mk_].mean()), int(mk_.sum())))
        cal = dict(brier=brier, brier_base=brier_base, reliability=rel)
    k = int(s.sum())
    lo, hi = wilson(k, n)
    return dict(cal=cal, rate=k / n, lo=lo, hi=hi, wrong=float(info["wrong"].mean()),
                hazard=float(info["hazard"].mean()),
                steps=float(info["steps"][s].mean()) if k else float("nan"),
                bat=float(info["bat"][s].mean() / E.BATTERY_S) if k else float("nan"))


T, N = args.T, args.envs * D
logf = open(os.path.join(OUT, "log.csv"), "w")
logf.write("iter,steps,train_succ,ent,eval_rate,eval_lo,eval_hi,eval_wrong,eval_hazard,eval_steps,gate,beats\n")
best = -1.0
t0 = time.time()
tr_succ = tr_eps = 0
last_ts = 0.0
for it in range(1, args.iters + 1):
    bl = np.zeros((T, N, E.P, E.P, 4), np.float32)
    bg = np.zeros((T, N, E.G, E.G, 1), np.float32)
    bv = np.zeros((T, N, E.NVEC), np.float32)
    ba = np.zeros((T, N), np.int64)
    blp = np.zeros((T, N), np.float32)
    bval = np.zeros((T, N), np.float32)
    br = np.zeros((T, N), np.float32)
    bd = np.zeros((T, N), np.float32)
    blab = np.zeros((T, N), np.float32)
    bmsk = np.zeros((T, N), np.float32)
    bpf = np.zeros((T, N), np.float32)
    bsucc = np.zeros((T, N), np.float32)
    for t in range(T):
        o = prep(env)
        bl[t], bg[t], bv[t] = o
        a, lp, v, pf = act_fn(*o)
        ba[t], blp[t], bval[t], bpf[t] = a.numpy(), lp.numpy(), v.numpy(), pf.numpy()
        _, r, d, info = env.step(ba[t], autoreset=True)
        br[t], bd[t] = r, d
        m = info["done"].astype(bool)
        if m.any():
            tr_eps += int(m.sum())
            tr_succ += int(info["succ"][m].sum())
            for e_i in np.nonzero(m)[0]:
                bsucc[t, e_i * D:(e_i + 1) * D] = float(info["succ"][e_i])
    last_v = value_fn(*prep(env)).numpy()
    if args.aux:
        # basari olasiligi: "basari" bir odul gibi (gamma~1) TD(lambda) hedefiyle ogrenilir; sonda kendi tahminiyle onyuklenir
        last_pf = pfind_fn(*prep(env)).numpy()
        ga = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nvp = last_pf if t == T - 1 else bpf[t + 1]
            nd = 1.0 - bd[t]
            delta = bsucc[t] + 0.999 * nvp * nd - bpf[t]
            ga = delta + 0.999 * args.lam * nd * ga
            blab[t] = np.clip(ga + bpf[t], 0.0, 1.0)
        bmsk[:] = 1.0
    adv = np.zeros((T, N), np.float32)
    gae = np.zeros(N, np.float32)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else bval[t + 1]
        nd = 1.0 - bd[t]
        delta = br[t] + args.gamma * nv * nd - bval[t]
        gae = delta + args.gamma * args.lam * nd * gae
        adv[t] = gae
    ret = adv + bval
    flat = lambda x: x.reshape((T * N,) + x.shape[2:])
    FL, FG, FV, FA, FLP, FADV, FRET, FLAB, FMSK = map(flat, (bl, bg, bv, ba, blp, adv, ret, blab, bmsk))
    FADV = (FADV - FADV.mean()) / (FADV.std() + 1e-8)
    idx = np.arange(T * N)
    for _ in range(args.epochs):
        np.random.shuffle(idx)
        for s0 in range(0, T * N, args.mb):
            j = idx[s0:s0 + args.mb]
            pl, vl, ent = train_step(FL[j], FG[j], FV[j], FA[j], FLP[j], FADV[j], FRET[j], FLAB[j], FMSK[j])
    if it % 5 == 0 or it == 1:
        ts = tr_succ / max(1, tr_eps)
        last_ts = ts
        print(f"it {it:4d} adim {it*T*N/1e6:.2f}M egitim-bulma %{100*ts:.1f} ent {float(ent):.2f} "
              f"[{time.time()-t0:.0f} s]", flush=True)
        tr_succ = tr_eps = 0
    if it % args.eval_every == 0 or it == args.iters:
        r = evaluate()
        gate = int(r["lo"] >= 0.60 and r["hi"] >= args.base)
        beats = int(r["lo"] > args.base_hi)
        print(f"  DEGERLENDIRME it {it}: bulma %{100*r['rate']:.1f} (GA {100*r['lo']:.1f}-{100*r['hi']:.1f}) "
              f"yanlis {r['wrong']:.2f} tehlike {r['hazard']:.1f} adim {r['steps']:.0f} "
              f"kapi {'GECTI' if gate else '-'} baz-cizgiyi-gecti {'EVET' if beats else '-'}", flush=True)
        logf.write(f"{it},{it*T*N},{last_ts:.4f},{float(ent):.4f},{r['rate']:.4f},{r['lo']:.4f},"
                   f"{r['hi']:.4f},{r['wrong']:.3f},{r['hazard']:.2f},{r['steps']:.1f},{gate},{beats}\n")
        logf.flush()
        if r["rate"] > best:
            best = r["rate"]
            model.save_weights(os.path.join(OUT, "best.weights.h5"))
        model.save_weights(os.path.join(OUT, "last.weights.h5"))
        sync_back()
logf.close()
if args.aux:
    import json
    r = evaluate(calib=True)
    with open(os.path.join(OUT, "kalibrasyon.json"), "w") as f:
        json.dump(r["cal"], f, indent=1)
    c = r["cal"]
    print(f"KALIBRASYON: Brier {c['brier']:.4f} (sabit tahmin {c['brier_base']:.4f}); guvenilirlik "
          f"(tahmin, gozlenen, n): " + "; ".join(f"{a:.2f},{b:.2f},{n}" for a, b, n in c["reliability"]), flush=True)
sync_back()
print("bitti", flush=True)
