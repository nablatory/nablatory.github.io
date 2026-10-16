# Contributing

Nablatory is a static site. Every number it shows is computed in the visitor's
browser by the engine in `src/engine`. Contributions are welcome as long as
they keep that property: nothing displayed may be fabricated, approximated
without saying so, or fetched from a server at run time.

## Setting up

```sh
npm ci
npm run dev        # http://localhost:5173
npm test           # numerical tests (vitest)
npm run check      # typecheck + lint + tests
```

Node 22 or newer. Regenerating the PyTorch fixtures needs Python 3.11+ with
`torch`: `python scripts/fixtures/generate_torch.py`.

## Adding a model

A model is one `ModelDef` object (`src/models/types.ts`) in
`src/models/<family>/<model>.ts`, listed in that family's `index.ts`. The
catalog, the model page, the route and the comparison options are all
generated from it. Read `src/models/convolutional/lenet5.ts` and
`src/models/feedforward/mlp.ts` first; they use every part of the contract.

### The contract

| Field | Meaning |
|---|---|
| `id` | URL slug, `/m/<id>`. Lower case, hyphenated, unique. |
| `signature` | The defining equation, one line of TeX, shown in the catalog. |
| `fidelity` | `exact` only for the published architecture at the published size; `equations` when the paper fixes no size (the perceptron, Elman network, LSTM, GRU); otherwise `reduced`, with `scaleNote` stating exactly what was reduced. |
| `limitations` | Plain sentences describing what is simplified, missing or approximated. Be specific. |
| `configSchema` | Each field declares `affects`: `architecture` (parameter shapes change, so parameters are re-initialised), `forward` (same parameters, different computation) or `data`. Kinds: `int`, `float`, `select`, `bool`, and `text` (with an optional `validate` that returns a sentence for unusable input). |
| `validate` | Return a sentence explaining why a configuration is mathematically invalid (e.g. a spatial size that reaches zero), or `null`. |
| `init(config, rng)` | Declare every parameter through `ParamBuilder`, which records the initialiser. Use `b.fixed()` for non-trainable tensors (masks, positional encodings, normalised adjacency). |
| `datasets` | Deterministic synthetic generators, or small static subsets with a licence (`public/data/LICENSES.md`). |
| `forward(batch, config, ctx)` | Build the computation with engine ops inside `scope('name', …)` blocks. Return named `outputs`, a scalar `loss` when the batch has targets, and `metrics`. |
| `training` | `support: 'full' | 'component' | 'none'` with a `note` when not full. Multi-objective models (GANs, actor–critic) declare `phases`, each owning a subset of the parameters. A phase whose loss contains stop-gradient targets (TD learning) also declares `checkLoss`, which freezes the targets so a finite-difference check is meaningful. `afterStep(params, step, config)` runs after each update for non-gradient updates such as target-network averages. |
| `procedures` | Named computations beyond one forward pass: sampling, generation, rollouts, ODE solves. They run with gradients off and must be cancellable through `shouldStop()` in long loops. |
| `views` | Which output visualisations to draw, referring to tensors as `out:<name>`, `in:<name>` or `proc:<id>:<tensor>`. See `src/viz/views.tsx` for the shape each kind expects. |
| `math` | TeX keyed by scope name (or its stem without trailing digits: `block` matches `block3`). |
| `checkpoints` | Optional pretrained weights in `public/weights/`, produced by a script in `scripts/train/` with this engine. |
| `guide` | The Guide page text: `what` (the model in plain words), `watch` (what to look for while it trains), `views` (titles of the views to show, default the first two) and up to three `try` entries, each a `label`, a `note`, a configuration `patch` and optional `train` steps. Every sentence must hold for this implementation at this scale; measure the numbers you quote from the page's opening state (default settings, default checkpoint or seed 1). Quote procedures, settings and buttons by their exact labels (“Sample”), so the Guide can show the cards they name. `tests/guide.test.ts` checks the names and runs every suggested change. |

### Rules

* **Scopes are the architecture diagram.** Each top-level `scope()` becomes a
  block on the canvas; nested scopes and ops appear when it is expanded. Name
  scopes after the paper's notation.
* **Every op must have a correct VJP.** Prefer composing existing ops in
  `src/engine/ops`. A new primitive needs a vector–Jacobian product and a
  finite-difference test; family-specific primitives live in
  `src/models/<family>/ops.ts` and are built with `op()` from
  `src/engine/graph.ts`.
* **Randomness comes from the graph.** Use `graphRng()` (or `ctx.rng`) inside
  `forward` and `ctx.rng` inside procedures. Never `Math.random()`.
* **Reduced means reduced, not different.** A reduced model keeps the
  defining equations of the paper and says what was made smaller.
* **No fabricated outputs.** A view may only display tensors the engine
  computed. If something cannot run in a browser, document it in
  `limitations` and offer the component that can.
* **Keep it fast.** A forward pass at the default configuration should take
  well under 100 ms on a laptop CPU, and a training step under 200 ms.

### Tests

Each family has `tests/<family>.test.ts`. At minimum, for every model:

1. **Gradient check** — `new Session(model).gradCheck(undefined, 3)` must have
   `maxRelError < 1e-5` (float64 central differences). Use a small batch.
   If the loss is large or sums many terms, set `gradCheck: { eps: 1e-5 }` on
   the model so the in-browser check uses the same step.
2. **Shapes and parameter counts** — against the paper where the model is exact.
3. **Learning** — a short training run lowers the validation loss.
4. **Model-specific identities** — e.g. a causal mask gives zero attention to
   future tokens; DDIM with η = 1 reproduces the DDPM posterior variance;
   GRU with z = 1 copies its state.

### Pretrained weights

`scripts/train/<model>.ts` trains with the same `Session` class the browser
uses and writes `public/weights/<name>.{json,bin}` with provenance (dataset,
seed, steps, optimiser, metrics, engine revision). Keep files small (aim for
under 2 MB). Run with `npx tsx scripts/train/<model>.ts`.

## Style

Prettier formats the code (`npm run format`). Interface copy is sentence
case, plain and specific; see `docs/DESIGN.md` for the visual system.

## Privacy

Do not add analytics, remote fonts, CDNs, or any network request other than
loading the site's own static files.
