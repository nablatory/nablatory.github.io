"""Reference values from PyTorch for the engine's numerical tests.

Each case records inputs, the forward output, a random upstream gradient
R = ∂L/∂y and the resulting input gradients for L = Σ y ⊙ R.
Run:  python3 scripts/fixtures/generate_torch.py   (writes tests/fixtures/torch.json)
"""

import json
import pathlib

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
CASES = []


def t(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).requires_grad_(True)


def enc(x):
    x = x.detach()
    return {"shape": list(x.shape), "data": x.reshape(-1).tolist()}


def case(name, fn, inputs, attrs=None):
    y = fn(**inputs)
    r = torch.randn_like(y)
    (y * r).sum().backward()
    CASES.append(
        {
            "name": name,
            "attrs": attrs or {},
            "inputs": {k: enc(v) for k, v in inputs.items() if torch.is_tensor(v)},
            "output": enc(y),
            "upstream": enc(r),
            "grads": {
                k: enc(v.grad)
                for k, v in inputs.items()
                if torch.is_tensor(v) and v.grad is not None
            },
        }
    )


case("linear", lambda x, w, b: F.linear(x, w, b), dict(x=t(2, 3, 5), w=t(4, 5), b=t(4)))
case("matmul_batched", lambda a, b: a @ b, dict(a=t(2, 3, 4), b=t(2, 4, 5)))
case("matmul_shared", lambda a, b: a @ b, dict(a=t(2, 3, 4), b=t(4, 6)))
case("matmul_shared_left", lambda a, b: a @ b, dict(a=t(3, 4), b=t(2, 4, 5)))
case(
    "conv2d",
    lambda x, w, b: F.conv2d(x, w, b, stride=2, padding=1),
    dict(x=t(2, 3, 7, 6), w=t(4, 3, 3, 3), b=t(4)),
    {"stride": 2, "padding": 1},
)
case(
    "conv2d_dilated_grouped",
    lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=2, dilation=2, groups=2),
    dict(x=t(1, 4, 6, 6), w=t(6, 2, 3, 3), b=t(6)),
    {"stride": 1, "padding": 2, "dilation": 2, "groups": 2},
)
case(
    "conv_transpose2d",
    lambda x, w, b: F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1),
    dict(x=t(2, 3, 4, 4), w=t(3, 2, 3, 3), b=t(2)),
    {"stride": 2, "padding": 1, "outputPadding": 1},
)
case("max_pool2d", lambda x: F.max_pool2d(x, 3, 2, 1), dict(x=t(2, 2, 7, 7)), {"k": 3, "stride": 2, "padding": 1})
case("avg_pool2d", lambda x: F.avg_pool2d(x, 2), dict(x=t(2, 2, 6, 6)), {"k": 2})
case("upsample_nearest", lambda x: F.interpolate(x, scale_factor=2, mode="nearest"), dict(x=t(1, 2, 3, 3)))
case(
    "layer_norm",
    lambda x, w, b: F.layer_norm(x, (6,), w, b, 1e-5),
    dict(x=t(2, 3, 6), w=t(6), b=t(6)),
)
case(
    "rms_norm",
    lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w,
    dict(x=t(3, 5), w=t(5)),
)


def bn_train(x, w, b):
    return F.batch_norm(x, torch.zeros(3), torch.ones(3), w, b, training=True, momentum=0.1, eps=1e-5)


case("batch_norm_train", bn_train, dict(x=t(4, 3, 2, 2), w=t(3), b=t(3)))
rm = torch.randn(3)
rv = torch.rand(3) + 0.5


def bn_eval(x, w, b):
    return F.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5)


case("batch_norm_eval", bn_eval, dict(x=t(4, 3, 2, 2), w=t(3), b=t(3)), {"runningMean": rm.tolist(), "runningVar": rv.tolist()})
case("softmax_axis1", lambda x: torch.softmax(x, 1), dict(x=t(2, 4, 3)))
case("log_softmax", lambda x: torch.log_softmax(x, -1), dict(x=t(3, 5)))
case("logsumexp", lambda x: torch.logsumexp(x, -1), dict(x=t(3, 5)))
case("sum_axes", lambda x: x.sum((0, 2)), dict(x=t(2, 3, 4)))
case("mean_keepdim", lambda x: x.mean(1, keepdim=True), dict(x=t(2, 3, 4)))
case("max_axis", lambda x: x.max(1).values, dict(x=t(3, 5)))
case("transpose_perm", lambda x: x.permute(2, 0, 1), dict(x=t(2, 3, 4)))
case("concat_axis1", lambda a, b: torch.cat([a, b], 1), dict(a=t(2, 3, 2), b=t(2, 1, 2)))
case("pad", lambda x: F.pad(x, (1, 2, 0, 1)), dict(x=t(2, 3, 3)))
case("broadcast_mul", lambda a, b: a * b, dict(a=t(2, 3, 4), b=t(3, 1)))
case("broadcast_div", lambda a, b: a / b, dict(a=t(2, 3), b=(torch.rand(1, 3) + 0.5).requires_grad_(True)))
case("broadcast_sub", lambda a, b: a - b, dict(a=t(4), b=t(3, 1)))
for act, fn in [
    ("tanh", torch.tanh),
    ("sigmoid", torch.sigmoid),
    ("gelu", F.gelu),
    ("silu", F.silu),
    ("softplus", F.softplus),
    ("elu", F.elu),
    ("leakyRelu", lambda x: F.leaky_relu(x, 0.01)),
    ("relu", F.relu),
    ("exp", torch.exp),
]:
    case(f"act_{act}", lambda x, fn=fn: fn(x), dict(x=t(4, 5)), {"act": act})
case("log", lambda x: torch.log(x), dict(x=(torch.rand(3, 4) + 0.1).requires_grad_(True)))
case("sqrt", lambda x: torch.sqrt(x), dict(x=(torch.rand(3, 4) + 0.1).requires_grad_(True)))

ids = [3, 0, 3, 1, 4]
case("gather", lambda w: F.embedding(torch.tensor(ids), w), dict(w=t(5, 3)), {"ids": ids})

targets = [1, 0, 3]
case("cross_entropy", lambda z: F.cross_entropy(z, torch.tensor(targets)), dict(z=t(3, 4)), {"targets": targets})
case(
    "cross_entropy_smooth",
    lambda z: F.cross_entropy(z, torch.tensor(targets), label_smoothing=0.1),
    dict(z=t(3, 4)),
    {"targets": targets, "labelSmoothing": 0.1},
)
yt = torch.randn(3, 4)
case("mse", lambda p: F.mse_loss(p, yt), dict(p=t(3, 4)), {"target": enc(yt)})
bt = (torch.rand(3, 4) > 0.5).double()
case("bce_logits", lambda z: F.binary_cross_entropy_with_logits(z, bt), dict(z=t(3, 4)), {"target": enc(bt)})

mask = torch.triu(torch.ones(4, 4), 1)


def causal_attention(q, k, v):
    s = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    s = s.masked_fill(mask.bool(), float("-inf"))
    return torch.softmax(s, -1) @ v


case("causal_attention", causal_attention, dict(q=t(2, 4, 3), k=t(2, 4, 3), v=t(2, 4, 3)))


def gru_cell(x, h, wih, whh, bih, bhh):
    return torch._VF.gru_cell(x, h, wih, whh, bih, bhh)


case("gru_cell", gru_cell, dict(x=t(2, 3), h=t(2, 4), wih=t(12, 3), whh=t(12, 4), bih=t(12), bhh=t(12)))


def lstm_cell(x, h, c, wih, whh, bih, bhh):
    hn, cn = torch._VF.lstm_cell(x, (h, c), wih, whh, bih, bhh)
    return torch.cat([hn, cn], -1)


case("lstm_cell", lstm_cell, dict(x=t(2, 3), h=t(2, 4), c=t(2, 4), wih=t(16, 3), whh=t(16, 4), bih=t(16), bhh=t(16)))

# Optimizers: three steps on fixed gradients.
OPT = []
g_seq = [torch.randn(5) for _ in range(3)]
theta0 = torch.randn(5)
for kind, ctor in [
    ("sgd", lambda p: torch.optim.SGD([p], lr=0.1, weight_decay=0.01)),
    ("momentum", lambda p: torch.optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)),
    ("nesterov", lambda p: torch.optim.SGD([p], lr=0.1, momentum=0.9, nesterov=True)),
    ("rmsprop", lambda p: torch.optim.RMSprop([p], lr=0.01, alpha=0.99, eps=1e-8)),
    ("adam", lambda p: torch.optim.Adam([p], lr=0.01, weight_decay=0.01)),
    ("adamw", lambda p: torch.optim.AdamW([p], lr=0.01, weight_decay=0.1)),
]:
    p = theta0.clone().requires_grad_(True)
    opt = ctor(p)
    traj = []
    for g in g_seq:
        p.grad = g.clone()
        opt.step()
        traj.append(p.detach().tolist())
    OPT.append({"kind": kind, "theta0": theta0.tolist(), "grads": [g.tolist() for g in g_seq], "trajectory": traj})

# Global-norm clipping.
gc = [torch.randn(3), torch.randn(2)]
clip_params = [torch.zeros(3, requires_grad=True), torch.zeros(2, requires_grad=True)]
for p, g in zip(clip_params, gc):
    p.grad = g.clone()
torch.nn.utils.clip_grad_norm_(clip_params, 0.5)
CLIP = {"grads": [g.tolist() for g in gc], "max": 0.5, "clipped": [p.grad.tolist() for p in clip_params]}

out = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "torch.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"torch": torch.__version__, "cases": CASES, "optimizers": OPT, "clip": CLIP}))
print(f"wrote {len(CASES)} cases, {len(OPT)} optimizer trajectories to {out}")
