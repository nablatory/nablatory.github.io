# Nablatory

A laboratory for the mathematics of neural networks that runs entirely in the
browser. Open a model, see its computation graph, run it on real input, read
any intermediate tensor and its gradient, change a weight or a configuration,
take one optimisation step and see exactly what moved.

Every number on screen is computed on the visitor's device by the engine in
this repository: forward passes, backward passes and training. There is no
server, no account and no analytics. Uploaded drawings, text and tensors never
leave the page.

## What is in it

Thirty-one models in eleven families. Each model page has two levels on the
same computation:

* **Guide** (`/m/<id>`) explains the model in plain words, shows its main
  outputs and loss while it trains, draws the network, and offers up to three
  suggested changes that re-run from the page's starting state.
* **Lab** (`/m/<id>/lab`) shows every setting, the full computation graph
  with each tensor and gradient, every model-specific output, the training
  timeline step by step, parameter editing, the gradient check, and the notes
  on equations, scale and limitations.

Switching between the two keeps the session: weights, training and outputs
carry over. The families:

| Family | Models |
|---|---|
| Feedforward | Perceptron, MLP, ResMLP |
| Convolutional | LeNet-5, VGG, ResNet, U-Net |
| Recurrent | RNN, LSTM, GRU |
| Transformers | encoder–decoder Transformer, BERT, GPT, ViT |
| Autoencoders | autoencoder, denoising autoencoder, VAE |
| Diffusion | DDPM, DDIM |
| Graph networks | GCN, GraphSAGE, GAT |
| GANs | GAN, DCGAN |
| World models | RSSM, Dreamer, TD-MPC |
| State space models | linear SSM, Mamba-style selective SSM |
| Neural dynamics | neural ODE, latent ODE |

Models are reduced in size where a browser requires it; each page says what
was reduced and keeps the published equations. `docs/MODELS.md` lists every
model with its scale, training support, pretrained weights and limitations.

The site has three sections:

* **Models** — the catalog, a live multilayer perceptron on the home page, and
  the two levels of every model page.
* **Compare** — two models or two configurations side by side on the same
  data and seed: metrics, loss and gradient-norm curves, block-by-block
  differences.
* **How it works** (`/about`) — how the numbers are verified, a one-click gradient check of every
  model in your browser, and a WebGPU matrix-multiply benchmark.

## How the numbers are checked

* **Reference tests.** Every engine operation and its gradient is compared with
  PyTorch 2.8 in float64 (`tests/fixtures`, relative error below 10⁻¹⁰), and
  whole computations (convolution → pooling → cross-entropy, attention with
  layer normalisation) with TensorFlow.js.
* **Gradient checks.** Every model's backpropagated gradients are compared with
  float64 central differences, in evaluation and in training mode. The same
  check runs in the browser from each model's Lab and from the How it works
  page.
* **Behavioural tests.** Each family's tests check model-specific identities
  (a causal mask gives zero attention to the future; DDIM with η = 1 reproduces
  the DDPM posterior variance; a GRU with z = 1 copies its state) and that
  short training runs learn. `tests/guide.test.ts` checks that every Guide
  names real views, settings and buttons, and that each suggested change is a
  valid configuration that trains with a finite loss.
* **Checkpoints.** Every file in `public/weights/` was trained with this engine
  by a script in `scripts/train/` and records its dataset, seed, steps,
  optimiser and final metrics. A test loads each one and re-measures its
  quality on the data shipped with the site.

## Run it

Requires Node 22 or later.

```sh
npm ci
npm run dev          # http://localhost:5173
npm test             # all numerical tests
npm run check        # lint, typecheck, tests
npm run build        # static site in dist/
npx vite preview     # serve dist/ at http://localhost:4173
```

## Repository

```
src/engine/      tensors, define-by-run autodiff, ops with VJPs, optimisers, gradient check
src/models/      one folder per family; each model is a ModelDef (src/models/types.ts)
src/runtime/     Session (shared by the worker, the tests and the training scripts), Web Worker, WebGPU
src/app/         pages, routing, experiment controller and workspace
src/viz/         architecture canvas, charts, tensor and model-specific views
src/ui/          design tokens, controls, styles
scripts/         checkpoint training, data fetching, route emission, docs
tests/           numerical, model and reference tests
docs/            ARCHITECTURE.md, DESIGN.md, MODELS.md, RELEASE.md
```

`CONTRIBUTING.md` explains how to add a model. `docs/ARCHITECTURE.md` explains
the technical decisions, `docs/DESIGN.md` the visual system, and
`docs/RELEASE.md` how to publish the site to GitHub Pages.

## Licence

Code: MIT (`LICENSE`). Shipped data: see `public/data/LICENSES.md`.
