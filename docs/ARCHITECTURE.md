# Nablatory — Architecture

Nablatory is a static, browser-only laboratory for neural network mathematics.
Every number shown in the interface is produced by the engine in this
repository, running on the visitor's device.

## 1. Name

GitHub organization `nablatory`, site `https://nablatory.github.io`.
The name joins ∇ (nabla, the gradient operator) and *laboratory*.

Candidates checked on 2026‑10‑14 (GitHub user/org namespace, npm, PyPI, web):

| Name | Result |
|---|---|
| nablab | taken on GitHub |
| gradscope | free, but too close to *Gradescope* |
| gradfield | free on GitHub, taken on PyPI |
| kernelscope, jacobilab, tensorlathe | free; weaker meaning |
| **nablatory** | free on GitHub/npm/PyPI, no product conflict found |

The organization has not been created. Publication steps are in
`docs/RELEASE.md` and require the owner's authorization.

## 2. Computational backend decision

| Option | Gradients | Graph editing | Trace of every intermediate | Verdict |
|---|---|---|---|---|
| ONNX Runtime Web | no (inference only) | no | partial | rejected |
| Transformers.js | no | no | no | rejected |
| TensorFlow.js | yes (`tf.grads`) | via re-tracing | upstream gradients of inner ops are not exposed | used as a **test reference** only |
| Custom tape engine (`src/engine`) | yes, reverse mode | yes | yes, every node keeps value and ∂L/∂value | **selected** |

Reasons: the product's central feature is inspecting the *actual* upstream and
downstream gradients of any selected operation. A small define‑by‑run tape over
`Float32Array` gives this directly, is deterministic with seeded RNG, has no
runtime dependency, and runs identically in the browser, in Web Workers, in
Node tests, and in the offline training scripts that produce shipped weights.

Correctness is established, not assumed:

* forward/backward fixtures generated with **PyTorch 2.8** (`scripts/fixtures/`)
  and committed as JSON; the engine is compared against them in Vitest;
* whole-computation cross‑checks against **TensorFlow.js CPU** (`tests/tfjs-reference.test.ts`);
* central finite‑difference gradient checks (float64 shadow evaluation) for
  every differentiable op.

### WebGPU

All tape operations execute on the CPU in a dedicated Web Worker. The compact
models used here (≤ ~1 M parameters) are dominated by per-op dispatch cost,
and gradient tracing needs synchronous access to every intermediate, whereas
reading a GPU buffer back requires awaiting. WebGPU is therefore detected and
reported in the top bar, and a WGSL tiled matrix-multiply kernel
(`src/runtime/webgpu.ts`) is offered as a benchmark on the How it works page, where it
is timed against the CPU kernel and its result compared element-wise with the
CPU result. It is not used in forward or backward passes. A GPU path for large
no-grad evaluations (decision fields, sampling) is possible future work; it
would have to keep the same validation.

## 3. Layers

```
src/engine     tensors, seeded RNG, ops with VJPs, tape autodiff, nn layers,
               losses, optimizers, finite-difference checks   (no DOM)
src/models     model definitions + datasets, one folder per family (no DOM)
src/runtime    worker protocol, sessions, training loop, trace serialization
src/state      tiny external store (useSyncExternalStore)
src/persist    IndexedDB experiments, versioned export/import
src/ui         design system: tokens, type, controls
src/viz        visualization primitives (canvas/SVG) driven by trace records
src/app        shell, routes, Explore / Experiment / Compare pages
scripts        route emission, fixture generation, offline training
tests          numerical tests
```

Only `src/engine` performs arithmetic on model tensors. Visualizations read
trace records; they never compute model quantities themselves (display
transforms such as colormaps and histograms excepted, and labelled).

## 4. Engine

* `Tensor` — `{shape, data: Float32Array}`, row‑major, NCHW for images.
* `Var` — tape node: value, optional grad, op name, attributes, parents,
  `scope` path (`block1/attn/q`) and stable `key` used for diffs across runs.
* Ops implement `forward` and a vector–Jacobian product. `backward(loss)` walks
  the tape in reverse creation order, accumulating ∂L/∂node for **every** node,
  so the inspector can show upstream gradients of internal operations.
* `noGrad(fn)` disables recording for sampling loops.
* Optimizers: SGD, SGD+momentum (heavy‑ball / Nesterov), RMSProp, Adam, AdamW.
  Gradient clipping by global norm or value. Weight decay is L2 for SGD/Adam,
  decoupled for AdamW.

## 5. Model definition contract

A model is one `ModelDef` object (`src/models/types.ts`); `CONTRIBUTING.md`
documents every field. In short: identity, family, references, `fidelity`
(`exact`, `equations` or `reduced`, with a `scaleNote`), limitations, a
`configSchema` whose fields declare whether they change parameter shapes, the
computation, or the data; `init(config, rng)`; `datasets`; `forward(batch, config, ctx)` returning
named outputs, a loss and metrics; `training` (`full`, `component` or `none`,
optionally with several `phases` that each own a parameter subset); no-grad
`procedures`; `views`; `math` keyed by scope; optional `checkpoints`; and
`guide`, the plain-language text of the model's Guide page.

The catalog (`src/models/catalog.ts`) lists families explicitly; each family's
`index.ts` lists its models. Pages, routes and comparison options are generated
from it.

## 6. Runtime

The UI talks to one worker through a typed message protocol. A *session* holds
config, parameters, optimizer state and the last trace. Commands: `create`,
`run`, `editParam`, `step`, `train` (multi‑step, yields between steps to accept
`pause`/`cancel`), `procedure`, `gradcheck`, `export`. Long loops yield every
step; a stuck computation can be stopped by terminating and re‑creating the
worker. Trace buffers returned by the worker are transferred; request payloads
are copied.

Diffs between runs match nodes by `key`. Comparisons use a shared colour scale.

## 7. Routing and deployment

Path routes (`/`, `/explore`, `/m/<model-id>`, `/compare`, `/about`).
After `vite build`, `scripts/emit-routes.ts` writes an `index.html` copy into
each route directory so direct loads return HTTP 200 on GitHub Pages; a
`404.html` copy covers unknown paths. GitHub Actions builds, tests and deploys.

## 8. Data and weights

Shipped weights are produced by `scripts/train/*.ts` using this engine, with
provenance (dataset, seed, steps, metrics) stored beside the weights. Datasets
are either generated deterministically or are small subsets with licences
listed in `public/data/LICENSES.md`. Nothing typed, uploaded or computed leaves
the browser; there is no analytics.

## 9. Visual system

See `docs/DESIGN.md`.
