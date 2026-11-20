"""Politika agi. w: genislik carpani (budama/karsilastirma icin), aux: elle eklenen basari-olasiligi head'i."""
import tensorflow as tf


def build(E, w=1.0, aux=False):
    L = tf.keras.layers
    c = lambda n: max(4, int(round(n * w)))
    loc = tf.keras.Input((E.P, E.P, 4))
    glo = tf.keras.Input((E.G, E.G, 1))
    vec = tf.keras.Input((E.NVEC,))
    a = L.Conv2D(c(16), 3, strides=2, padding="same", activation="relu", name="l_c1")(loc)
    a = L.Conv2D(c(16), 3, strides=2, padding="same", activation="relu", name="l_c2")(a)
    a = L.Dense(c(64), activation="relu", name="l_d")(L.Flatten()(a))
    g = L.Conv2D(c(8), 3, strides=2, padding="same", activation="relu", name="g_c1")(glo)
    g = L.Conv2D(c(8), 3, strides=2, padding="same", activation="relu", name="g_c2")(g)
    g = L.Dense(c(32), activation="relu", name="g_d")(L.Flatten()(g))
    v = L.Dense(c(32), activation="relu", name="v_d")(vec)
    h = L.Concatenate()([a, g, v])
    h = L.Dense(c(128), activation="relu", name="h1")(h)
    h = L.Dense(c(64), activation="relu", name="h2")(h)
    outs = [L.Dense(E.N_ACT, name="logits")(h), L.Dense(1, name="value")(h)]
    if aux:
        outs.append(L.Dense(1, name="p_find")(h))
    return tf.keras.Model([loc, glo, vec], outs)
