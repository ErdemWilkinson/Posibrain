"""Politikayi TFLite int8'e cevirir ve simulasyonda float ile int8'i ayni tohumlarda karsilastirir.
Kullanim: python quant.py --weights runs/t1b_1km/best.weights.h5 --out runs/t1b_1km [--width 1.0] [--drones 1] [--cells 100] [--aux 0]
Cikti: <out>/policy_int8.tflite, <out>/quant.json (boyut KB, float ve int8 bulma orani + %95 GA)."""
import os
import json
import argparse
import importlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # nicemleme CPU'da yeter, egitimdeki GPU/RAM'i bolme
import numpy as np
import tensorflow as tf
from net import build
from baselines import wilson

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--width", type=float, default=1.0)
ap.add_argument("--drones", type=int, default=1)
ap.add_argument("--cells", type=int, default=100)
ap.add_argument("--aux", type=int, default=0)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed0", type=int, default=2_000_000, help="test tohumlari (egitim/secimde kullanilmayan)")
args = ap.parse_args()
D = max(1, args.drones)
E = importlib.import_module("envm" if D > 1 else "env")

full = build(E, args.width, bool(args.aux))
full.load_weights(args.weights)
outs = [full.outputs[0]] + ([full.outputs[2]] if args.aux else [])  # logits (+ p_find); deger head'i dagitimda gerekmez
deploy = tf.keras.Model(full.inputs, outs)


def mk(n):
    return E.SearchEnv(n, cells=args.cells, drones=D) if D > 1 else E.SearchEnv(n, cells=args.cells)


def prep(e):
    return (np.ascontiguousarray(e.loc.transpose(0, 2, 3, 1)), e.glo[..., None].copy(), e.vec.copy())


# temsilci veri: float politika oynarken gorulen gozlemler
rep_env = mk(128)
rep = []
for t in range(120):
    o = prep(rep_env)
    rep.append(o)
    a = np.argmax(deploy(list(o), training=False)[0] if args.aux else deploy(list(o), training=False), axis=1)
    rep_env.step(a, autoreset=True)
rep_idx = [(t, i) for t in range(0, 120, 4) for i in range(0, rep[0][0].shape[0], 16)]


def rep_gen():
    names = deploy.input_names  # loc, glo, vec sirasi; ada gore ver ki donusturucu siralamayi karistirmasin
    for t, i in rep_idx[:400]:
        yield {names[0]: rep[t][0][i:i + 1], names[1]: rep[t][1][i:i + 1], names[2]: rep[t][2][i:i + 1]}


conv = tf.lite.TFLiteConverter.from_keras_model(deploy)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tfl = conv.convert()
os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, "policy_int8.tflite")
open(path, "wb").write(tfl)
print(f"int8 boyut: {len(tfl)/1024:.1f} KB  ({sum(int(np.prod(v.shape)) for v in deploy.trainable_variables)} parametre)", flush=True)

interp = tf.lite.Interpreter(model_content=tfl)
inp = interp.get_input_details()
outd = interp.get_output_details()
order = {d["name"]: d for d in inp}
# girdi sirasi: loc(15,15,4), glo(20,20,1), vec(NVEC) -> sekle gore esle
def slot(shape):
    for d in inp:
        if tuple(d["shape"][1:]) == shape:
            return d
    raise RuntimeError(f"girdi yok {shape}")


s_loc, s_glo, s_vec = slot((E.P, E.P, 4)), slot((E.G, E.G, 1)), slot((E.NVEC,))
logit_det = [d for d in outd if d["shape"][-1] == E.N_ACT][0]


def tfl_logits(o):
    n = o[0].shape[0]
    for d, x in ((s_loc, o[0]), (s_glo, o[1]), (s_vec, o[2])):
        interp.resize_tensor_input(d["index"], [n] + list(x.shape[1:]))
    interp.allocate_tensors()
    for d, x in ((s_loc, o[0]), (s_glo, o[1]), (s_vec, o[2])):
        interp.set_tensor(d["index"], x.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(logit_det["index"])


def run(policy):
    n = args.n
    e = mk(n)
    e.reset(np.ones(n, bool), seeds=np.arange(args.seed0, args.seed0 + n))
    info = None
    for _ in range(E.MAX_STEPS + 5):
        if e.done.all():
            break
        a = np.argmax(policy(prep(e)), axis=1)
        _, _, _, info = e.step(a)
    k = int(info["succ"].sum())
    lo, hi = wilson(k, n)
    return dict(rate=k / n, lo=lo, hi=hi, wrong=float(info["wrong"].mean()), hazard=float(info["hazard"].mean()))


def float_logits(o):
    r = full(list(o), training=False)
    return r[0].numpy()


rf = run(float_logits)
ri = run(tfl_logits)
res = dict(size_kb=len(tfl) / 1024, float=rf, int8=ri)
json.dump(res, open(os.path.join(args.out, "quant.json"), "w"), indent=1)
print(f"float: %{100*rf['rate']:.1f} (GA {100*rf['lo']:.1f}-{100*rf['hi']:.1f})   "
      f"int8: %{100*ri['rate']:.1f} (GA {100*ri['lo']:.1f}-{100*ri['hi']:.1f})", flush=True)
