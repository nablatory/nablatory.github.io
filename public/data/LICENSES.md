# Data licences

Third-party data shipped under `public/data/`.

## Tiny Shakespeare (`text/shakespeare.txt`)

- Source: A. Karpathy, char-rnn, `data/tinyshakespeare/input.txt`,
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Content: passages from Shakespeare's plays. The text is in the public domain.
- Shipped: the first 299,868 characters of the 1,115,394-character file (cut at a
  blank line), unmodified. Produced by `scripts/data/fetch-text.ts`.
- Used by: `gpt` (character-level language model). The first 90 % is the
  training split, the last 10 % the validation split.

## MNIST (`mnist/*.u8`)

- Source: Y. LeCun, C. Cortes and C. J. C. Burges, *The MNIST database of
  handwritten digits*. Downloaded from the PyTorch mirror
  https://ossci-datasets.s3.amazonaws.com/mnist/ (byte-identical to the
  original IDX files) by `scripts/data/fetch-mnist.ts`.
- Licence: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), as the
  dataset is commonly distributed. The subset is redistributed unmodified under
  the same licence.
- Shipped: the first 6,000 training images and labels and the first 1,000 test
  images and labels, with the IDX headers removed (raw 28 × 28 bytes, row-major;
  labels one byte each).
- Used by: `lenet-5`, `vgg`, `resnet`, `vit`, `autoencoder`,
  `denoising-autoencoder`, `vae`, `dcgan` and the checkpoints trained on them.

## Zachary's karate club (in `src/models/data/graphs.ts`)

- Source: W. W. Zachary, "An information flow model for conflict and fission in
  small groups", *Journal of Anthropological Research* 33(4):452–473, 1977.
- Content: 34 members and 78 friendship ties, plus the faction each member
  joined after the split. These are published scientific facts, reproduced from
  the paper's adjacency matrix and indexed as in NetworkX's `karate_club_graph`.
- Used by: `gcn`, `graphsage`, `gat`.

## Generated in the browser

All other datasets (two moons, XOR, spirals, digit sequences, the agreement
grammar, synthetic shapes, the point-mass environment, the pendulum, the
stochastic block model and the toy distributions for diffusion and GANs) are
produced by deterministic, seeded code in `src/models/data/` and carry no
third-party licence.
