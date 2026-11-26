"""Probe 4: encoder cost and permutation invariance, numpy reference (torch is NOT installed in .venv).

For each candidate block: exact parameter count, analytic MACs, numpy single-thread latency (batch 1,
the eval setting), and op count (to estimate torch-eager per-op overhead). Also correctness tests:
  - entity encoders: output unchanged under a joint permutation of (features, type, mask) of real slots
  - output unchanged when padded slots are filled with garbage
  - n = 0 (the most common case) gives finite output
  - slicing to the n real entities gives the same output as the padded 32-slot pass
  - ray circular conv: circularly shifting the 64 rays circularly shifts the stride-1 feature map

Run: .venv/bin/python research/spatial_probes/probe_encoders.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

RNG = np.random.default_rng(0)
F32 = np.float32
OPS = [0]  # op counter (array ops per forward, ~ number of torch kernel launches)


def P(*shape, scale=None):
    scale = scale or 1.0 / np.sqrt(shape[0] if len(shape) > 1 else 1)
    return (RNG.standard_normal(shape) * scale).astype(F32)


class Lin:
    def __init__(self, i, o):
        self.W, self.b = P(i, o), np.zeros(o, F32)
        self.macs_per_row = i * o

    def __call__(self, x):
        OPS[0] += 2
        return x @ self.W + self.b

    @property
    def n(self):
        return self.W.size + self.b.size


def relu(x):
    OPS[0] += 1
    return np.maximum(x, 0)


def layernorm(x):
    OPS[0] += 4
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5)


def masked_softmax(s, mask):  # s (..., n), mask (n,) bool
    OPS[0] += 4
    s = np.where(mask, s, -1e9)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s) * mask
    return e / np.maximum(e.sum(-1, keepdims=True), 1e-9)


N_TYPES, T_EMB, N_FEAT, D = 12, 8, 5, 64


def ent_input(feats, etype, emb):
    OPS[0] += 2
    prox = 1.0 / (1.0 + feats[:, :1] * 8.0)  # distance/1024 -> 1/(1+d/128): resolves the near range
    return np.concatenate([feats, prox, emb[etype]], 1)  # (n, 5+1+8 = 14)


IN = N_FEAT + 1 + T_EMB


class DeepSets:
    name = "masked DeepSets (phi 14-64-64, mean+max pool, rho 129-64)"

    def __init__(self):
        self.emb = P(N_TYPES, T_EMB, scale=0.1)
        self.l1, self.l2, self.r = Lin(IN, D), Lin(D, D), Lin(2 * D + 1, D)

    def params(self):
        return self.emb.size + self.l1.n + self.l2.n + self.r.n

    def macs(self, n):
        return n * (self.l1.macs_per_row + self.l2.macs_per_row) + self.r.macs_per_row

    def __call__(self, feats, etype, mask):
        m = mask.astype(bool)
        h = relu(self.l2(relu(self.l1(ent_input(feats, etype, self.emb)))))
        cnt = m.sum()
        OPS[0] += 5
        mean = (h * m[:, None]).sum(0) / max(cnt, 1)
        mx = np.where(m[:, None], h, 0.0).max(0) if len(m) else np.zeros(D, F32)  # h>=0 so 0 is a safe floor
        return relu(self.r(np.concatenate([mean, mx, [cnt / 32.0]]).astype(F32)))


class AttnPool:
    name = "attention pooling (phi 14-64-64, 4-head PMA, 1 learned query + learned null token)"

    def __init__(self, heads=4):
        self.emb = P(N_TYPES, T_EMB, scale=0.1)
        self.l1, self.l2 = Lin(IN, D), Lin(D, D)
        self.k, self.v, self.o = Lin(D, D), Lin(D, D), Lin(D, D)
        self.q = P(D, scale=0.1)
        self.null = P(D, scale=0.1)  # always-present token: empty set -> finite, learned "nothing seen" output
        self.h = heads

    def params(self):
        return self.emb.size + self.l1.n + self.l2.n + self.k.n + self.v.n + self.o.n + self.q.size + self.null.size

    def macs(self, n):
        return n * (self.l1.macs_per_row + self.l2.macs_per_row) + (n + 1) * (2 * D * D + 2 * D) + D * D

    def pool(self, h, m):
        OPS[0] += 2
        h = np.concatenate([self.null[None], h], 0)
        m = np.concatenate([[True], m])
        K, V = self.k(h), self.v(h)
        dh = D // self.h
        Kh, Vh, qh = K.reshape(-1, self.h, dh), V.reshape(-1, self.h, dh), self.q.reshape(self.h, dh)
        OPS[0] += 3
        s = np.einsum("hd,nhd->hn", qh, Kh) / np.sqrt(dh)
        a = masked_softmax(s, m)
        out = np.einsum("hn,nhd->hd", a, Vh).reshape(D)
        return relu(self.o(out))

    def __call__(self, feats, etype, mask):
        m = mask.astype(bool)
        h = relu(self.l2(relu(self.l1(ent_input(feats, etype, self.emb)))))
        return self.pool(h, m)


class TinyTransformer(AttnPool):
    name = "tiny transformer (embed 14-64, 2 x [4-head self-attn + FFN 128], pre-LN) + attention pooling"

    def __init__(self, layers=2, heads=4):
        super().__init__(heads)
        self.emb_in = Lin(IN, D)
        self.layers = [(Lin(D, 3 * D), Lin(D, D), Lin(D, 2 * D), Lin(2 * D, D)) for _ in range(layers)]

    def params(self):
        n = self.emb.size + self.emb_in.n + self.k.n + self.v.n + self.o.n + self.q.size + self.null.size
        return n + sum(sum(l.n for l in L) for L in self.layers)

    def macs(self, n):
        per = n * (3 * D * D + D * D + 2 * D * 2 * D) + 2 * n * n * D
        return n * IN * D + len(self.layers) * per + (n + 1) * (2 * D * D + 2 * D) + D * D

    def __call__(self, feats, etype, mask):
        m = mask.astype(bool)
        x = self.emb_in(ent_input(feats, etype, self.emb))
        dh = D // self.h
        for qkv, o, f1, f2 in self.layers if len(x) else []:  # empty set: nothing to self-attend
            q, k, v = np.split(qkv(layernorm(x)), 3, axis=1)
            OPS[0] += 6
            s = np.einsum("nhd,mhd->hnm", q.reshape(-1, self.h, dh), k.reshape(-1, self.h, dh)) / np.sqrt(dh)
            a = masked_softmax(s, m)
            y = np.einsum("hnm,mhd->nhd", a, v.reshape(-1, self.h, dh)).reshape(-1, D)
            x = x + o(y)
            x = x + f2(relu(f1(layernorm(x))))
        return self.pool(x, m)


class CircConvRays:
    name = "rays: circular Conv1d 6-32 k5 s1, 32-32 k5 s2, 32-64 k3 s2 -> flatten 1024 -> 128"

    def __init__(self):
        self.emb = P(6, 4, scale=0.3)  # hit-type embedding (4) instead of one-hot (6)
        self.c = [(P(5 * 6, 32), 1), (P(5 * 32, 32), 2), (P(3 * 32, 64), 2)]
        self.fc = Lin(16 * 64, 128)

    def params(self):
        return self.emb.size + sum(w.size + w.shape[1] for w, _ in self.c) + self.fc.n

    def macs(self):
        L, tot = 64, 0
        for w, s in self.c:
            L = L // s
            tot += L * w.shape[0] * w.shape[1]
        return tot + self.fc.macs_per_row

    def features(self, dist, hit):
        OPS[0] += 3
        x = np.concatenate([dist[:, None], (1.0 / (1.0 + dist * 16.0))[:, None], self.emb[hit]], 1).astype(F32)  # (64, 6)
        for w, s in self.c:
            k = w.shape[0] // x.shape[1]
            p = k // 2
            xp = np.concatenate([x[-p:], x, x[:p]], 0)  # circular padding
            cols = np.lib.stride_tricks.sliding_window_view(xp, (k, x.shape[1]))[::s, 0].reshape(-1, k * x.shape[1])
            OPS[0] += 4
            x = relu(cols @ w)
        return x

    def __call__(self, dist, hit):
        return relu(self.fc(self.features(dist, hit).reshape(-1)))


class GRUCell:
    def __init__(self, i, h):
        self.Wx, self.Wh = Lin(i, 3 * h), Lin(h, 3 * h)
        self.hd = h

    def params(self):
        return self.Wx.n + self.Wh.n

    def macs(self):
        return self.Wx.macs_per_row + self.Wh.macs_per_row

    def __call__(self, x, h):
        gx, gh = self.Wx(x), self.Wh(h)
        H = self.hd
        OPS[0] += 8
        r = 1 / (1 + np.exp(-(gx[:H] + gh[:H])))
        z = 1 / (1 + np.exp(-(gx[H:2 * H] + gh[H:2 * H])))
        n = np.tanh(gx[2 * H:] + r * gh[2 * H:])
        return (1 - z) * n + z * h


class MapCNN:
    name = "ego-map CNN: C x 32 x 32 -> conv3x3 s2 16 -> conv3x3 s2 32 -> conv3x3 s2 32 -> 512 -> 128"

    def __init__(self, C=6):
        self.c = [P(9 * C, 16), P(9 * 16, 32), P(9 * 32, 32)]
        self.fc = Lin(4 * 4 * 32, 128)
        self.C = C

    def params(self):
        return sum(w.size + w.shape[1] for w in self.c) + self.fc.n

    def macs(self):
        S, tot = 32, 0
        for w in self.c:
            S //= 2
            tot += S * S * w.shape[0] * w.shape[1]
        return tot + self.fc.macs_per_row

    def __call__(self, x):  # (32, 32, C)
        for w in self.c:
            xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
            cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3, x.shape[2]))[::2, ::2, 0].reshape(-1, 9 * x.shape[2])
            OPS[0] += 4
            x = relu(cols @ w).reshape(x.shape[0] // 2, x.shape[1] // 2, -1)
        return relu(self.fc(x.reshape(-1)))


class DepthCNN:
    name = "full depth CNN: 1x120x160 -> conv8x8 s4 16 -> conv4x4 s2 32 -> conv3x3 s2 32 -> 6x8x32=1536 -> 128"

    def __init__(self):
        self.c = [(P(64, 16), 8, 4), (P(16 * 16, 32), 4, 2), (P(9 * 32, 32), 3, 2)]
        self.fc = Lin(6 * 8 * 32, 128)

    def shapes(self):
        H, W, out = 120, 160, []
        for w, k, s in self.c:
            H, W = (H - k) // s + 1, (W - k) // s + 1
            out.append((H, W))
        return out

    def params(self):
        return sum(w.size + w.shape[1] for w, _, _ in self.c) + self.fc.n

    def macs(self):
        return sum(h * w_ * c[0].shape[0] * c[0].shape[1] for (h, w_), c in zip(self.shapes(), self.c)) + self.fc.macs_per_row

    def __call__(self, x):  # (120, 160, 1)
        for w, k, s in self.c:
            cols = np.lib.stride_tricks.sliding_window_view(x, (k, k, x.shape[2]))[::s, ::s, 0]
            h, w_ = cols.shape[:2]
            OPS[0] += 4
            x = relu(cols.reshape(h * w_, -1) @ w).reshape(h, w_, -1)
        return relu(self.fc(x.reshape(-1)))


class DepthRows:
    name = "depth eye-level band: rows 56-63 mean -> 160 -> avgpool4 -> 40 -> Linear 40-64"

    def __init__(self):
        self.fc = Lin(40, 64)

    def params(self):
        return self.fc.n

    def macs(self):
        return 8 * 160 + self.fc.macs_per_row

    def __call__(self, depth):
        OPS[0] += 3
        row = depth[56:64].mean(0).reshape(40, 4).mean(1) / 255.0
        return relu(self.fc(row.astype(F32)))


def bench(fn, reps=2000):
    for _ in range(50):
        fn()
    OPS[0] = 0
    fn()
    ops = OPS[0]
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e6
    return {"median_us": float(np.median(ts)), "p99_us": float(np.quantile(ts, 0.99)), "ops": ops}


def random_entities(n, garbage=False):
    f = np.zeros((32, 5), F32)
    t = np.zeros(32, np.int64)
    m = np.zeros(32, np.int8)
    ang = RNG.uniform(-np.pi / 4, np.pi / 4, n)
    f[:n] = np.stack([RNG.uniform(0, 2, n), np.sin(ang), np.cos(ang), RNG.normal(0, 2, n), RNG.normal(0, 2, n)], 1)
    t[:n] = RNG.integers(1, 12, n)
    m[:n] = 1
    if garbage:
        f[n:] = RNG.normal(0, 100, (32 - n, 5))
        t[n:] = RNG.integers(0, 12, 32 - n)
    return f, t, m


def invariance_tests(enc):
    res = {}
    worst_perm = worst_pad = worst_slice = 0.0
    finite0 = True
    for n in (0, 1, 2, 5, 16, 32):
        f, t, m = random_entities(n)
        base = enc(f, t, m)
        finite0 &= bool(np.isfinite(base).all())
        for _ in range(5):
            perm = np.concatenate([RNG.permutation(n), np.arange(n, 32)])
            worst_perm = max(worst_perm, float(np.abs(enc(f[perm], t[perm], m[perm]) - base).max()))
            full = RNG.permutation(32)  # mixes padded slots in among real ones
            worst_perm = max(worst_perm, float(np.abs(enc(f[full], t[full], m[full]) - base).max()))
        fg, tg, mg = random_entities(n, garbage=True)
        fg[:n], tg[:n] = f[:n], t[:n]
        worst_pad = max(worst_pad, float(np.abs(enc(fg, tg, mg) - base).max()))
        worst_slice = max(worst_slice, float(np.abs(enc(f[:n], t[:n], m[:n]) - base).max()))
    res["max_abs_diff_permuted"] = worst_perm
    res["max_abs_diff_garbage_padding"] = worst_pad
    res["max_abs_diff_sliced_to_n"] = worst_slice
    res["finite_all_n_incl_0"] = finite0
    return res


if __name__ == "__main__":
    out = {"numpy": np.__version__, "threads": os.environ["OPENBLAS_NUM_THREADS"], "entity": {}, "other": {}}
    for enc in (DeepSets(), AttnPool(), TinyTransformer()):
        r = {"params": int(enc.params()), "macs_n32": int(enc.macs(32)), "macs_n2": int(enc.macs(2))}
        r.update(invariance_tests(enc))
        for n in (32, 16, 2, 0):
            f, t, m = random_entities(n)
            r[f"lat_padded32_n{n}"] = bench(lambda: enc(f, t, m))
        for n in (2, 16):
            f, t, m = random_entities(n)
            fs, ts_, ms = f[:n], t[:n], m[:n]
            r[f"lat_sliced_n{n}"] = bench(lambda: enc(fs, ts_, ms))
        out["entity"][enc.name] = r
        print(enc.name, json.dumps(r), flush=True)
    rays = CircConvRays()
    dist, hit = RNG.uniform(0, 1, 64).astype(F32), RNG.integers(0, 6, 64)
    a = rays.c[0]
    rays_eq = None
    # equivariance: first conv layer is stride 1 -> shifting input by k shifts its output by k
    w1 = rays.c
    rays.c = [w1[0]]
    f0 = rays.features(dist, hit)
    f1 = rays.features(np.roll(dist, 3), np.roll(hit, 3))
    rays_eq = float(np.abs(np.roll(f0, 3, axis=0) - f1).max())
    rays.c = w1
    out["other"]["rays"] = {"name": rays.name, "params": rays.params(), "macs": rays.macs(),
                            "shift_equivariance_err_layer1": rays_eq, "lat": bench(lambda: rays(dist, hit))}
    gru = GRUCell(256, 256)
    h = np.zeros(256, F32)
    x = RNG.standard_normal(256).astype(F32)
    out["other"]["gru256"] = {"params": gru.params(), "macs": gru.macs(), "lat": bench(lambda: gru(x, h))}
    trunk = Lin(128 + 64 + 64 + 128, 256)
    heads = Lin(256, 9 + 1 + 16)
    z = RNG.standard_normal(384).astype(F32)
    out["other"]["trunk_384_256"] = {"params": trunk.n, "macs": trunk.macs_per_row, "lat": bench(lambda: relu(trunk(z)))}
    out["other"]["heads_256_26"] = {"params": heads.n, "macs": heads.macs_per_row, "lat": bench(lambda: heads(h))}
    mp = MapCNN(6)
    grid = RNG.standard_normal((32, 32, 6)).astype(F32)
    out["other"]["map_cnn"] = {"name": mp.name, "params": mp.params(), "macs": mp.macs(), "lat": bench(lambda: mp(grid), 1000)}
    dc = DepthCNN()
    dep = RNG.uniform(0, 1, (120, 160, 1)).astype(F32)
    out["other"]["depth_cnn"] = {"name": dc.name, "params": dc.params(), "macs": dc.macs(), "shapes": dc.shapes(), "lat": bench(lambda: dc(dep), 300)}
    dr = DepthRows()
    dep2 = RNG.integers(0, 255, (120, 160)).astype(np.uint8)
    out["other"]["depth_rows"] = {"name": dr.name, "params": dr.params(), "macs": dr.macs(), "lat": bench(lambda: dr(dep2))}
    for k, v in out["other"].items():
        print(k, json.dumps(v), flush=True)
    from pathlib import Path
    d = Path(__file__).resolve().parents[2] / "runs" / "spatial-intelligence-researcher"
    d.mkdir(parents=True, exist_ok=True)
    (d / "encoders.json").write_text(json.dumps(out, indent=1))
