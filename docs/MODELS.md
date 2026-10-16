# Models

Generated from `src/models/catalog.ts` by `scripts/catalog-doc.ts`. Do not edit by hand.

## Feedforward

Compositions of affine maps and pointwise nonlinearities.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Perceptron](../src/models/feedforward) `perceptron` | published equations | full | — | [F. Rosenblatt 1958](https://doi.org/10.1037/h0042519) |
| [Multilayer perceptron](../src/models/feedforward) `mlp` | published equations | full | — | [D. E. Rumelhart et al. 1986](https://doi.org/10.1038/323533a0) |
| [Residual MLP](../src/models/feedforward) `resmlp` | reduced | full | — | [K. He et al. 2016](https://arxiv.org/abs/1603.05027) |

**Perceptron.** A single linear threshold unit. It separates the plane with one line, and cannot represent XOR.

- The learning rule is expressed as gradient descent on the perceptron criterion; with batch size 1 and SGD at learning rate 1 this is exactly Rosenblatt’s update w ← w + t·x on misclassified points.
- At t·s = 0 the subgradient 0 is used, so a perceptron initialised exactly at zero never moves. Initialisation is therefore random.
- Rosenblatt’s 1958 perceptron had a fixed, randomly wired layer of association units between the sensory inputs and the response unit; this is the single response unit with learned weights on its inputs, the simple perceptron analysed by Minsky and Papert (1969).

**Multilayer perceptron.** Affine maps alternating with a pointwise nonlinearity, trained by backpropagation to separate two classes in the plane. *Scale:* Rumelhart, Hinton and Williams (1986) define the architecture and the learning rule but no canonical size; depth and width here are chosen for two-dimensional inputs.

- Two-dimensional synthetic data only, so that the full decision function can be drawn.
- Softmax output with cross-entropy, trained with Adam; the 1986 paper used logistic output units with a squared-error criterion and gradient descent with momentum.

**Residual MLP.** Fully connected blocks that add their output to their input, so each block learns a correction to the identity. *Scale:* Fully connected residual blocks in the pre-activation ordering of He et al. (2016): the normalisation precedes the weights and the sum is not followed by a nonlinearity. Layer normalisation replaces batch normalisation, and each block has one normalisation and one nonlinearity (LN → W₁ → σ → W₂) rather than the paper’s two; depth and width are chosen for two-dimensional inputs.

- Two-dimensional synthetic data only. Turning off the skip connection keeps all parameters and only removes the addition, so the same weights can be compared with and without it.

## Convolutional

Shared local filters slid across a spatial grid.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [LeNet-5](../src/models/convolutional) `lenet-5` | published size | full | yes | [Y. LeCun et al. 1998](https://doi.org/10.1109/5.726791) |
| [VGG](../src/models/convolutional) `vgg` | reduced | full | yes | [K. Simonyan, A. Zisserman 2015](https://arxiv.org/abs/1409.1556) |
| [ResNet](../src/models/convolutional) `resnet` | reduced | full | yes | [K. He et al. 2016](https://arxiv.org/abs/1512.03385) |
| [U-Net](../src/models/convolutional) `unet` | reduced | full | yes | [O. Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597) |

**LeNet-5.** Seven-layer convolutional network for 32×32 digit images: alternating convolution and subsampling, then fully connected layers. *Scale:* Layer sizes, C3 connection table, trainable subsampling and the scaled tanh are as published. The default output layer is a linear softmax classifier; the published Euclidean RBF output is selectable (its fixed codes are random ±1 vectors rather than the paper’s 7×12 character bitmaps).

- Trained here with Adam and cross-entropy, not the stochastic diagonal Levenberg–Marquardt method used in the paper.
- The shipped checkpoint was trained on the full MNIST training set by scripts/train/lenet5.ts; browser training uses the 6,000-image subset.

**VGG.** Stacks of 3×3 convolutions with ReLU, 2×2 max-pooling between stages and channel doubling, followed by a fully connected classifier. *Scale:* Published VGG-16 (configuration D): 224×224 RGB input, 5 stages of 2–3 convolutions with 64–512 channels, FC-4096, FC-4096, FC-1000, 138 M parameters. Here: 28×28 greyscale MNIST, 3 stages of 2 convolutions with 6, 12 and 24 channels, FC-128, FC-128, FC-10 (55,760 parameters at the defaults). Kernel size, padding, max-pooling, channel doubling and the three-layer head with dropout 0.5 are as published.

- Trained with Adam on the 6,000-image MNIST subset, not SGD with momentum, weight decay and learning-rate drops on ImageNet.
- No multi-scale training, scale jittering, horizontal flips or dense multi-crop evaluation.
- The batch-norm variant (VGG-BN) postdates the paper; it places batch norm between each convolution and its ReLU and removes the convolution bias, which β makes redundant.
- Channels double at every stage without the paper’s cap at 512.
- Convolutions and the first two FC layers start from He normal, the last FC layer from Glorot uniform, all biases from zero; the paper drew weights from a zero-mean Gaussian with variance 10⁻² and zero biases, and initialised its deeper configurations from a trained configuration A.
- The shipped checkpoint was trained offline on the full 60,000-image MNIST training set by scripts/train/vgg.ts; browser training samples the 6,000-image subset, and the validation metric uses the first 200 test images.

**ResNet.** Residual network: each block learns a residual F(x) that is added to an identity (or projection) shortcut, so information and gradients can bypass the convolutions. *Scale:* Follows the CIFAR-10 ResNets of the paper (Section 4.2): 3×3 stem, three stages of basic blocks with widths w, 2w, 4w and strides 1, 2, 2, global average pooling and one FC layer, with option A shortcuts. Published: w = 16 and n = 3–18 blocks per stage (ResNet-20 to ResNet-110, 0.27–1.7 M parameters) on 32×32 RGB images. Here: w = 6 and n = 1 at the defaults (8 weighted layers, 10,852 trainable parameters) on 28×28 MNIST.

- Trained with Adam on the 6,000-image MNIST subset, not SGD with momentum 0.9, weight decay 10⁻⁴ and learning-rate drops on CIFAR-10 or ImageNet.
- No data augmentation (the paper pads by 4 and takes random 32×32 crops and flips).
- Option A appends the zero channels after the existing ones rather than splitting them on both sides; this only permutes channels.
- In the pre-activation variant the stem is a bare convolution and every block, including the first, applies its BN–ReLU inside the residual branch only, so identity and projection shortcuts read the raw block input x. He et al. (2016b) place the first BN–ReLU right after the stem, before the split, and let projection shortcuts read the pre-activated signal.
- Only basic two-convolution blocks; the bottleneck blocks used for ResNet-50 and deeper are not included.
- The shipped checkpoint was trained offline on the full 60,000-image MNIST training set by scripts/train/resnet.ts; browser training samples the 6,000-image subset, and the validation metric uses the first 200 test images.

**U-Net.** Encoder–decoder for per-pixel classification: a contracting path of convolutions and pooling, an expanding path of up-convolutions, and skip connections that concatenate encoder features into the decoder at the same resolution. *Scale:* Published: 572×572 input tiles, 4 poolings, 64–1024 channels, unpadded convolutions with cropped skip connections, 388×388 output, 31 M parameters. Here: 32×32 tiles, 2 poolings, 8–32 channels (29,330 parameters at the defaults) on synthetic shapes with 2 classes (3 optional). Padded convolutions are the default, so the output has the input size; the paper’s unpadded convolutions with centre-cropped skips are selectable and need a larger tile.

- The data are synthetic ellipses and rectangles, not microscopy images; there is no elastic-deformation augmentation.
- The paper’s pixel-wise loss weight map w(x), which emphasises thin borders between touching cells, is not used; all pixels have weight 1.
- In the unpadded variant the label map is the centre crop of a larger generated scene; the paper instead mirrors the image at its borders (overlap-tile strategy), which is not implemented.
- Trained with Adam rather than SGD with momentum 0.99 and batch size 1.
- The Dice term is an optional addition (Milletari et al., 2016), not part of the original U-Net.
- With 3 classes the network must tell ellipses from rectangles by their outlines, which this small model learns slowly: an offline run of the default size (1,800 Adam steps, batch 8) reached a mean shape-class IoU of only 0.28 on held-out scenes, so no 3-class checkpoint is shipped. The 2-class checkpoint reached 0.96.

## Recurrent

A hidden state updated by the same cell at every time step.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Elman recurrent network](../src/models/recurrent) `rnn` | published equations | full | — | [J. L. Elman 1990](https://doi.org/10.1207/s15516709cog1402_1) |
| [Long short-term memory](../src/models/recurrent) `lstm` | published equations | full | — | [S. Hochreiter, J. Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.8.1735) |
| [Gated recurrent unit](../src/models/recurrent) `gru` | published equations | full | — | [K. Cho et al. 2014](https://arxiv.org/abs/1406.1078) |

**Elman recurrent network.** One hidden state updated from the previous state and the current input by the same affine map and nonlinearity at every step. *Scale:* The Elman update with PyTorch’s parameterisation (two bias vectors) and initialisation. Hidden size and tasks are small so it trains in the browser.

- Single layer, unidirectional, trained by full backpropagation through time on short synthetic sequences.
- The ReLU option uses the default initialisation, not the identity initialisation of Le et al. (2015), so it can diverge without gradient clipping.
- Elman’s original network fed back a copy of the hidden layer through fixed one-to-one context units, used logistic units, and was trained by ordinary backpropagation with the context units treated as inputs, so no error flowed back through time; this is the now-standard equivalent with a learned W_hh, trained through every step.

**Long short-term memory.** A memory cell updated additively and guarded by input, forget and output gates, so error signals can cross many steps without vanishing. *Scale:* The LSTM with forget gate (Gers et al. 2000) in PyTorch’s layout and initialisation, without peephole connections. Hidden size and tasks are small so it trains in the browser.

- Single layer, unidirectional, trained by full backpropagation through time. The 1997 paper truncated the gradient at the cell; here it flows through every gate.
- No peephole connections (Gers & Schmidhuber 2000) and no projection layer.

**Gated recurrent unit.** A hidden state that is interpolated between its previous value and a candidate, with an update gate z and a reset gate r and no separate memory cell. *Scale:* The GRU in PyTorch’s formulation (reset applied after the recurrent product, as in cuDNN) and initialisation. Hidden size and tasks are small so it trains in the browser.

- Single layer, unidirectional, trained by full backpropagation through time on short synthetic sequences.
- Cho et al. (2014) apply the reset gate before the recurrent product, n = tanh(W x + U(r ⊙ h)); this uses the PyTorch/cuDNN variant n = tanh(W x + r ⊙ (U h + b)), which differs slightly.
- Cho et al. write h_t = z ⊙ h_{t−1} + (1 − z) ⊙ h̃ with the same convention as here; some texts swap the roles of z and 1 − z.

## Transformers

Tokens mixed by softmax attention, then transformed position-wise.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Transformer (encoder–decoder)](../src/models/transformer) `transformer` | reduced | full | — | [A. Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) |
| [BERT (masked language model)](../src/models/transformer) `bert` | reduced | full | — | [J. Devlin et al. 2019](https://arxiv.org/abs/1810.04805) |
| [GPT (character level)](../src/models/transformer) `gpt` | reduced | full | yes | [A. Radford et al. 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| [Vision Transformer (ViT)](../src/models/transformer) `vit` | reduced | full | yes | [A. Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929) |

**Transformer (encoder–decoder).** The original attention-only sequence-to-sequence model: an encoder of self-attention layers and a decoder that also attends to the encoder output, here learning to reverse or sort digit strings. *Scale:* Reduced from the base model (N = 6 encoder and 6 decoder layers, d_model = 512, 8 heads, d_ff = 2048, 37,000 shared BPE tokens, WMT translation) to N = 2 + 2 layers, d_model = 32, 4 heads, d_ff = 4·d_model = 128, an 11-symbol vocabulary (digits and <s>) and sequences of 6 digits. Sinusoidal encodings, √d embedding scaling, tied embeddings and post-LN residual blocks are as published.

- A synthetic digit task instead of machine translation; sequences have a fixed length, so no padding mask, end-of-sequence symbol or beam search is needed.
- Trained with constant-rate Adam; the paper’s warm-up schedule (lr ∝ d^{−0.5}·min(step^{−0.5}, step·warmup^{−1.5})) and label smoothing (ε = 0.1) are not used.
- Dropout applies to sublayer outputs and to the embedding sums, as in the paper, and also to the attention weights (as in the reference implementation; the paper does not mention it). It is 0 by default because the task has unlimited fresh data.
- In the attention maps rows are query positions and columns key positions. Encoder positions are labelled with the source digits and decoder positions with the decoder input (<s>, y_1, …, y_{L−1}), so decoder row t is the query that predicts y_{t+1}.

**BERT (masked language model).** A bidirectional transformer encoder pre-trained by selecting 15 % of the tokens, replacing most of them with [MASK], and predicting them from both sides, here on generated sentence pairs with number agreement. *Scale:* Reduced from BERT-base (12 layers, d = 768, 12 heads, d_ff = 3,072, 512 positions, 30,522 WordPiece tokens, BooksCorpus + Wikipedia) to 2 layers, d = 32, 4 heads, d_ff = 128, 16 positions and a 37-symbol synthetic vocabulary. Embeddings (token + learned position + segment, then LayerNorm), post-LN GELU blocks, the masking rule and the tied MLM output layer are as published.

- Masked language modelling only: the next-sentence-prediction objective and its [CLS] classifier are omitted (RoBERTa showed it is not needed). Segment embeddings are kept.
- Masks are drawn afresh for every training batch (dynamic masking, as in RoBERTa); BERT duplicated its data ten times with static masks.
- A sequence in which no token is selected gets one selected position, so the effective selection rate is slightly above the configured rate on these short sentences.
- The corpus is a generated toy language. Nouns, adjectives and verb stems are chosen at random, so even a perfect model cannot predict them; only the agreement-bearing words are predictable.
- Weights start from N(0, 0.02²) without truncation; LayerNorm uses ε = 10⁻⁵ instead of 10⁻¹².

**GPT (character level).** A decoder-only transformer trained to predict the next character of Shakespeare, with a causal mask so each position sees only its past. *Scale:* Reduced from GPT-2 small (12 layers, d = 768, 12 heads, 1,024-token context, 50,257 BPE tokens, ~40 GB WebText) to 2 layers, d = 64, 4 heads, a 64-character context and a 65-character vocabulary, trained on ~270 KB of Shakespeare. Block structure (pre-LN, GELU, learned positions, final LayerNorm, tied unembedding) follows GPT-2.

- Characters, not byte-pair tokens: the model must learn spelling, so text is word-like rather than fluent at this size.
- Only ~300 KB of text is shipped. The checkpoint saw 1.6 million characters (about six passes over the training split) and reaches 1.71 nats (2.47 bits) per character on the held-out 10 %; samples have the layout of a play and many real short words, but few coherent phrases.
- GPT-2’s 1/√N scaling of residual projections at initialisation is omitted; weights start from N(0, 0.02²) as in GPT-1, except the position embeddings, which start from N(0, 0.01²) as in the GPT-2 code.
- Sampling recomputes the whole context for every new character (no key–value cache), which is exact but slower.
- With the mask set to “none” every position except the last in the window can read the character it is asked to predict; after a few hundred steps the loss falls towards zero (from the checkpoint it first jumps to about 3.7) without the model learning to generate. This setting exists to show what the mask prevents.

**Vision Transformer (ViT).** An image cut into patches, each patch linearly embedded as a token, a learned [CLS] token prepended, and a transformer encoder classifying from the [CLS] output. *Scale:* Reduced from ViT-B/16 (224×224 RGB images, 16×16 patches → 196 tokens, 12 layers, D = 768, 12 heads, MLP 3,072, pre-trained on ImageNet-21k or JFT-300M) to 28×28 MNIST digits with 7×7 patches → 16 tokens, 3 layers, D = 32, 4 heads, MLP 128, trained from scratch on MNIST.

- No pre-training: the shipped checkpoint was trained from scratch on the 60,000 MNIST training images by scripts/train/vit-mnist.ts, and browser training uses the 6,000-image subset. ViT’s published results depend on large-scale pre-training; without it ViTs underperform convolutional networks of similar size, and this one reaches 95.96 % MNIST test accuracy against 98.29 % for the LeNet-5 checkpoint.
- The classification head is a single linear layer on the final [CLS] state (the paper’s fine-tuning head); the one-hidden-layer pre-training head is not used.
- No data augmentation and no weight decay. Browser training uses constant-rate Adam without warm-up; the shipped checkpoint used a 100-step linear warm-up and cosine decay (stated in its provenance).
- The attention-from-[CLS] image shows weights of the last layer averaged over heads. It shows where [CLS] reads from, not which pixels the prediction depends on.

## Autoencoders

An encoder to a low-dimensional code and a decoder back.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Autoencoder](../src/models/autoencoder) `autoencoder` | reduced | full | — | [D. E. Rumelhart et al. 1986](https://doi.org/10.7551/mitpress/5236.001.0001) |
| [Denoising autoencoder](../src/models/autoencoder) `denoising-autoencoder` | reduced | full | — | [P. Vincent et al. 2008](https://doi.org/10.1145/1390156.1390294) |
| [Variational autoencoder](../src/models/autoencoder) `vae` | reduced | full | yes | [D. P. Kingma, M. Welling 2014](https://arxiv.org/abs/1312.6114) |

**Autoencoder.** An encoder maps a digit to a short code z and a decoder maps z back to an image; both are trained to reproduce the input. *Scale:* Hinton & Salakhutdinov (2006) used a 784–1000–500–250–30 encoder with a mirrored decoder, pretrained layer by layer as RBMs. Here the encoder has one hidden layer (or two strided convolutions), the decoder mirrors it, and training is end to end with Adam from random initialisation.

- No RBM pretraining; the deep autoencoder of Hinton & Salakhutdinov is not reproduced at its published depth.
- The binary cross-entropy option treats grey-level intensities in [0, 1] as Bernoulli targets, as is common practice; it is not a normalised likelihood for continuous pixels.
- Browser training uses the 6,000-image MNIST subset; the latent plot uses the first 500 test images.

**Denoising autoencoder.** The input is corrupted at random and the network is trained to reconstruct the clean digit, so it must learn structure rather than copy pixels. *Scale:* One hidden layer with sigmoid units, tied weights, masking noise and cross-entropy as in Vincent et al. (2008). The paper used 1,000–2,000 hidden units, plain SGD, and stacked the layers before supervised fine-tuning; here the default is 128 units, Adam, and a single layer with no stacking.

- Single layer only; stacking and the supervised fine-tuning stage of the paper are not included.
- Grey-level intensities in [0, 1] are used as cross-entropy targets; the paper also used the [0, 1] scaled pixels.
- Salt-and-pepper noise is from the 2010 journal version; Gaussian noise is an option the 2008 paper discusses but did not use on MNIST.
- A fresh corruption is drawn for every training batch. Evaluation reuses one fixed draw (the same seed every time), so the validation loss measures the same corrupted digits at every check; it is a single sample of the expected loss, not its average over corruptions.

**Variational autoencoder.** The encoder outputs a Gaussian over codes, a sample is decoded through a Bernoulli likelihood, and training maximises the evidence lower bound. *Scale:* Gaussian MLP encoder and Bernoulli MLP decoder with one hidden layer each and one sample of ε per data point, as in Kingma & Welling (2014, appendix C). The paper used 500 tanh hidden units, Adagrad and minibatches of 100; the default here is 256 ReLU units (tanh is selectable) and Adam. β ≠ 1 gives the β-VAE objective of Higgins et al. (2017).

- Grey-level MNIST intensities are used directly as Bernoulli targets; the paper does the same, but this is not a normalised likelihood for continuous pixels, so the reported −ELBO is not comparable with numbers on dynamically binarised MNIST.
- The ELBO is a single-sample Monte Carlo estimate on every forward pass. ε is redrawn for every training batch; evaluation reuses one fixed draw, so the validation −ELBO is the same one-sample estimate at every check rather than an average over ε.
- No importance-weighted bound or marginal-likelihood estimate is computed.
- Browser training uses the 6,000-image MNIST subset; the shipped checkpoint was trained on the full 60,000-image training set.

## Diffusion

Data corrupted by Gaussian noise and recovered by a learned denoiser.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Denoising diffusion probabilistic model](../src/models/diffusion) `ddpm` | reduced | full | yes | [J. Ho et al. 2020](https://arxiv.org/abs/2006.11239) |
| [Denoising diffusion implicit model](../src/models/diffusion) `ddim` | reduced | full | yes | [J. Song et al. 2021](https://arxiv.org/abs/2010.02502) |

**Denoising diffusion probabilistic model.** Data is destroyed by T small Gaussian noising steps; a network trained to predict the added noise runs the chain backwards to generate new points. *Scale:* The forward process, the ε-prediction objective L_simple, the sinusoidal time embedding (applied to the rescaled time 1000·t/T) and ancestral sampling are as published. The data are two-dimensional point clouds instead of images, the denoiser is a residual MLP (30,530 parameters at the default size) instead of a U-Net with attention, and the default T is 100 instead of 1,000 (β rescaled by 1000/T).

- Two-dimensional toy data only, so that samples, noised data, trajectories and the learned score can be drawn exactly.
- The network is an MLP, not the U-Net of the paper; there is no attention, group normalisation or dropout.
- The variance is fixed (β_t or β̃_t); the learned variances and hybrid loss of Nichol & Dhariwal are not implemented.
- Sample quality is measured by the squared maximum mean discrepancy (Gaussian kernel, bandwidth 0.3; the biased estimator that includes the diagonal terms) against 400 data points, not by FID or likelihood.
- The shipped checkpoint was trained with T = 100 and the linear schedule. Changing T or the schedule keeps the weights, which then see noise levels they were not trained on.

**Denoising diffusion implicit model.** A non-Markovian sampler for a DDPM-trained noise predictor: it skips timesteps, and with η = 0 it is a deterministic map from noise to data. *Scale:* The DDIM update (Song et al., eq. 12), the η-family of noise scales (eq. 16) and evenly spaced timestep sub-sequences are as published. The noise predictor is the same two-dimensional residual MLP as the DDPM page, trained with the same L_simple objective; it is not an image U-Net.

- Training is identical to DDPM (DDIM changes only the sampler). This page trains its own copy of the network in the browser and loads the same offline checkpoint file as the DDPM page.
- Only evenly spaced sub-sequences τ are offered; the quadratic spacing used for CIFAR-10 in the paper is not.
- The predicted x̂₀ is not clipped (there is no natural data range for the toy distributions).
- Two-dimensional data only; sample quality is measured by MMD² against 400 data points, not FID.

## Graph neural networks

Node features updated by aggregating over neighbours.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Graph convolutional network](../src/models/graph) `gcn` | reduced | full | — | [T. N. Kipf, M. Welling 2017](https://arxiv.org/abs/1609.02907) |
| [GraphSAGE](../src/models/graph) `graphsage` | reduced | full | — | [W. L. Hamilton et al. 2017](https://arxiv.org/abs/1706.02216) |
| [Graph attention network](../src/models/graph) `gat` | reduced | full | — | [P. Veličković et al. 2018](https://arxiv.org/abs/1710.10903) |

**Graph convolutional network.** Each layer mixes every node’s features with its neighbours’ through the symmetrically normalised adjacency matrix, then applies a shared linear map. Trained semi-supervised from a few labelled nodes. *Scale:* The propagation rule, renormalisation trick, Glorot initialisation, masked cross-entropy and Adam at learning rate 0.01 for 200 steps are as published. The graphs are small (34 and 60 nodes) instead of Citeseer, Cora and Pubmed. The default network follows the paper’s karate-club illustration (Appendix A: three graph convolutions with tanh, hidden widths 4 and 2, a softmax layer on top, one labelled node per class) rather than the two-layer 16-unit ReLU model of the citation experiments (selectable); here the three layers have 16 and 2 hidden units with tanh, the third is the class layer, and two nodes per class are labelled. Dropout is 0.2 instead of 0.5, the L2 penalty 5 × 10⁻⁴ is applied to every parameter (biases included) through the optimiser rather than to the first layer’s weights only, and there is no early stopping.

- Transductive setting on one small graph: the whole graph is processed in every step (full-batch training), and the loss reads only the few labelled nodes.
- Dense N × N adjacency and propagation matrices; this is exact for these graph sizes but scales as O(N²) memory, unlike the sparse implementations used for large graphs.
- The karate-club labels are the factions after the club split, as recorded by Zachary (1977); features are one-hot node identities, so the model sees structure only.
- The normalised adjacency is recomputed from the adjacency input in every forward pass (it receives no gradient), so self-loops and the normalisation can be switched without re-initialising.
- Row normalisation D̃⁻¹Ã is offered for comparison; the paper uses the symmetric form.

**GraphSAGE.** Each node concatenates its own representation with an aggregate of its neighbours’ (mean or max-pooling), applies a shared linear map and a nonlinearity, and normalises the result to unit length. *Scale:* Algorithm 1 of the paper (aggregate, concatenate, linear map, ReLU, L2 normalisation) with K = 2 and the mean and max-pooling aggregators; here the second layer produces the class logits directly, without ReLU, normalisation or a separate classifier. Trained supervised on a few labelled nodes of a small graph (not the paper’s inductive Reddit, citation and PPI benchmarks), with 32 instead of 256 hidden units, on the full graph each step instead of minibatches of target nodes, and with the full neighbourhood by default instead of sampled neighbourhoods of 25 and 10.

- Transductive setting on one small graph: the whole graph is processed in every step (full-batch training), and the loss reads only the few labelled nodes.
- Dense N × N adjacency and propagation matrices; this is exact for these graph sizes but scales as O(N²) memory, unlike the sparse implementations used for large graphs.
- The karate-club labels are the factions after the club split, as recorded by Zachary (1977); features are one-hot node identities, so the model sees structure only.
- Neighbour sampling: with “sampled neighbours” = 0 (default) the aggregator reads the full neighbourhood. With k > 0, during training only, k neighbours per node are drawn uniformly with replacement from the seeded graph generator; evaluation and inspection always use the full neighbourhood, whereas the paper also samples at test time.
- The LSTM aggregator and the unsupervised random-walk loss are not implemented.
- Max pooling routes the gradient to the first maximal neighbour, as torch.max does.
- With one-hot identity features (karate club) the separate self weights give every node its own free parameters; with two labelled members per faction GraphSAGE generalises worse than GCN here (held-out accuracy about 0.5–0.75 across seeds after 200 steps, against about 0.93–0.97 for GCN). On the block model, whose features are shared, it reaches about 0.8–0.9.

**Graph attention network.** Each node attends over its neighbours: a learned score of every (node, neighbour) pair is normalised with a softmax over the neighbourhood and used to average the neighbours’ transformed features, with several heads in parallel. *Scale:* The attention mechanism, masking to N(i) ∪ {i}, LeakyReLU slope 0.2, ELU, Glorot initialisation, multi-head concatenation in hidden layers, head averaging in the output layer, and dropout on both layer inputs and on the normalised coefficients are as published. The default has 4 heads × 4 units and 2 averaged output heads on 34- and 60-node graphs, instead of 8 heads × 8 units with a single output head on Cora and Citeseer (8 averaged output heads on Pubmed); dropout is 0.2 instead of 0.6, the learning rate 0.01 instead of 0.005, the L2 penalty 5 × 10⁻⁴ is applied to every parameter (biases included) through the optimiser, and there is no early stopping.

- Transductive setting on one small graph: the whole graph is processed in every step (full-batch training), and the loss reads only the few labelled nodes.
- Dense N × N adjacency and propagation matrices; this is exact for these graph sizes but scales as O(N²) memory, unlike the sparse implementations used for large graphs.
- The karate-club labels are the factions after the club split, as recorded by Zachary (1977); features are one-hot node identities, so the model sees structure only.
- Attention is computed densely for all N² pairs and masked with −∞ outside each neighbourhood, so non-neighbours get exactly zero weight. The paper’s sparse implementation computes the same coefficients on edges only.
- Dropout is applied to the layer inputs and to the normalised coefficients, as in the paper; the shown coefficients are those before dropout.

## GANs

A generator and a discriminator trained against each other.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Generative adversarial network](../src/models/gan) `gan` | reduced | full | — | [I. Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661) |
| [DCGAN](../src/models/gan) `dcgan` | reduced | full | yes | [A. Radford et al. 2016](https://arxiv.org/abs/1511.06434) |

**Generative adversarial network.** A generator maps noise to points in the plane and a discriminator learns to tell them from samples of a target distribution; each is trained against the other. *Scale:* The objective, the alternating updates (k = 1) and both generator losses are those of Goodfellow et al. (2014). The paper trained on MNIST, TFD and CIFAR-10 with a maxout discriminator; here both networks are two-hidden-layer MLPs on a two-dimensional target so that the generated distribution and the discriminator can be drawn exactly.

- Two-dimensional targets only; the image-scale version is the DCGAN model.
- The discriminator uses leaky ReLU units rather than maxout, and no dropout.
- Both networks share one optimiser configuration (each phase keeps its own Adam moments).
- Mode coverage counts a mode as covered when at least 5 of the generated points lie within 3σ of its centre; it is a diagnostic for the ring and grid targets, not a standard metric.

**DCGAN.** A GAN whose generator is a stack of transposed convolutions with batch normalisation and whose discriminator is a strided convolutional network, here generating 28 × 28 digits. *Scale:* Follows the architecture guidelines of Radford et al. (2016): strided (transposed) convolutions instead of pooling, batch normalisation in both networks except the generator output and discriminator input, ReLU in G with tanh output, LeakyReLU(0.2) in D, weights from N(0, 0.02²), Adam with learning rate 2·10⁻⁴ and β₁ = 0.5. The paper generated 64 × 64 images with 1024→512→256→128 generator channels; here the output is 28 × 28 MNIST in [−1, 1] with g = 16 and d = 16 base channels by default, two (transposed) convolutions per network, and browser batches of 16 instead of 128 (the shipped checkpoint used 32).

- Two upsampling stages instead of four, with far fewer channels, so samples are coarser than in the paper.
- The noise is z ~ N(0, I) with 32 dimensions by default; the paper drew a 100-dimensional z from a uniform distribution.
- Batch normalisation uses batch statistics during training and running averages (momentum 0.1) when sampling; generated images can differ slightly between the two modes.
- The discriminator sees real and generated batches in separate passes, so its batch statistics are never mixed, as in the reference PyTorch implementation.
- No sample-quality score such as FID is computed; the training script records the losses and discriminator outputs it measured.
- Browser training uses the 6,000-image MNIST subset; the shipped checkpoint was trained on the full 60,000-image training set.

## World models

Learned latent dynamics used to predict and plan.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Recurrent state-space model (PlaNet)](../src/models/world-model) `rssm` | reduced | full | yes | [D. Hafner et al. 2019](https://arxiv.org/abs/1811.04551) |
| [Dreamer](../src/models/world-model) `dreamer` | reduced | full | yes | [D. Hafner et al. 2020](https://arxiv.org/abs/1912.01603) |
| [TD-MPC](../src/models/world-model) `td-mpc` | reduced | full | yes | [N. Hansen et al. 2022](https://arxiv.org/abs/2203.04955) |

**Recurrent state-space model (PlaNet).** A latent dynamics model learned from image sequences: a deterministic recurrent state carries memory, a Gaussian stochastic state captures what the next frame adds, and decoders reconstruct images and rewards. *Scale:* The RSSM structure, Gaussian prior and posterior with minimum σ 0.1, reparameterised sampling and the free-nats ELBO are as in PlaNet (with ELU activations, as in Dreamer, rather than PlaNet’s ReLU). Sizes are reduced: h has 32 GRU units instead of 200, z 6 dimensions instead of 30, the default encoder and decoder are two-layer MLPs (a reduced two-layer convolutional pair is selectable) instead of PlaNet’s four-layer convolutional networks, frames are 16 × 16 greyscale instead of 64 × 64 colour, sequences are 12 steps instead of 50, and free nats are 1 instead of 3.

- Trained offline on a fixed dataset from a scripted policy; PlaNet’s data-collection loop with the cross-entropy-method planner is not included (see TD-MPC for latent planning).
- Latent overshooting is not used; the loss is the one-step ELBO (the objective Dreamer later adopted).
- Free nats are applied per time step to the KL summed over latent dimensions, as max(KL, free nats); Dreamer v1’s reference implementation clips the batch-mean KL instead, which penalises every step once the average exceeds the threshold.
- The point-mass environment is a stand-in for the DeepMind Control Suite: its dynamics are simple and its images are 16 × 16.
- A few hundred browser steps lower the loss visibly but leave open-loop predictions blurred after a few steps. The shipped checkpoint (6,000 steps) reaches a per-pixel open-loop error of about 0.002 five steps and 0.010 fifteen steps after 5 context frames, averaged over the held-out episodes; predicting the background alone gives about 0.03.

**Dreamer.** An RSSM world model learned from image sequences, plus an actor and a critic trained entirely inside the model: the actor maximises λ-returns of imagined trajectories by backpropagating through the learned dynamics. *Scale:* The RSSM world model (as on the RSSM page), the tanh-Gaussian actor trained by analytic gradients through imagined dynamics, and the critic regressed on λ-returns follow Dreamer (v1), as do the three Adam optimisers (world model, actor, critic) that each step once per training step. Reduced: h 32 / z 6 units, MLP image encoder and decoder on 16 × 16 frames, one-hidden-layer actor and critic of 64 units, 32 imagination start states per batch instead of all, imagination horizon 10 instead of 15, discount 0.95 instead of 0.99, and one learning rate for all three optimisers.

- All three components train in the browser, but on a fixed offline dataset from a scripted policy: Dreamer’s loop of collecting new episodes with the current actor is not run, so the actor can exploit model errors in regions the data never visited.
- Each training step runs three phases on the same batch: world model, then actor (world-model and critic parameters frozen, gradients flow through them), then critic. As in the reference implementation, imagination starts from the posterior states of the world-model pass, and the critic regresses on the actor pass’s imagined states and λ-returns (gradients stopped). Unlike it, the phases run one after another, so the actor imagines with the world model already updated in this step. Each phase has its own Adam state.
- Imagination starts from 32 of the 96 posterior states of a batch (at the default 8 sequences of 12 steps; spread evenly and configurable up to all of them, which the paper uses) to keep a training step fast; the start states are detached.
- The actor is a tanh-squashed Gaussian as in Dreamer v1, but with σ = softplus(x + 0.54) + 0.1 (initial σ ≈ 1.1, minimum 0.1) and an unscaled mean, where the reference implementation uses an initial σ of 5, a minimum of 10⁻⁴ and the mean 5 tanh(μ/5). In imagination the actor reads the imagined state without the stop-gradient that the reference implementation places on its input, so the return gradient also flows through the actor’s own input features: the paper’s objective taken literally.
- No episode termination or learned discount; the actor has no entropy bonus, as in Dreamer v1 for continuous actions.
- The point-mass task is far simpler than the DeepMind Control Suite tasks of the paper. With a few hundred browser steps the actor improves its imagined return long before it improves in the simulator. The shipped checkpoint (5,000 steps) collects a mean return of 35.5 over 40 simulator steps from the held-out start states, against 14.8 for the scripted policy that produced the data (40 is the maximum).

**TD-MPC.** A task-oriented latent dynamics model without a decoder: an encoder, latent dynamics, reward and twin Q heads trained by latent consistency, reward and temporal-difference losses, used for planning with MPPI over short horizons and a learned value beyond them. *Scale:* Encoder h, latent dynamics d, reward R, twin Q, policy π, the temporally weighted reward + TD + consistency objective (c₁ = 0.5, c₂ = 0.1, c₃ = 2 and temporal weight ρ = 0.5, the paper’s λ) and MPPI planning (6 iterations, temperature 0.5, σ clipped to [0.05, 2], horizon 5) follow TD-MPC. Reduced: 16-dimensional latent and one-hidden-layer MLPs of 32 units instead of 50 and two hidden layers of 512, state observations of a point mass, discount 0.9 instead of 0.99, 64 + 4 candidates and 8 elites instead of 512 + 25 and 64, and no momentum on μ (paper 0.1).

- Trained offline on the fixed scripted-policy dataset; there is no replay buffer filled by the planner, no seed-step exploration and no prioritised replay.
- TD targets and consistency targets use the current networks with gradients stopped, not the slowly updated (exponential-moving-average) target networks of the paper.
- No layer normalisation in the Q networks and no gradient-scale tricks of the reference implementation.
- MPPI here executes the first action of the final mean μ^J. Algorithm 2 of the paper samples a ~ N(μ^J, (σ^J)²); the reference implementation draws one elite trajectory with probability proportional to its weight and adds σ noise during training. μ is refitted without momentum.
- Policy candidates for MPPI are one noise-free policy rollout plus rollouts with Gaussian noise of standard deviation 0.2 on the actions; the reference implementation adds noise of 0.05 to every policy trajectory and to π in the TD target, whereas the TD target here uses π(z⁺) without noise.
- The model is trained on 5-step windows, so long-horizon planning relies on the learned Q beyond them; after a few hundred browser steps Q is still inaccurate and closed-loop control is usually worse than the scripted policy. The shipped checkpoint (4,000 steps) reaches a mean closed-loop return of 37.4 over 40 simulator steps from the held-out start states, against 14.8 for the scripted policy (40 is the maximum).

## State space models

Linear recurrences from discretised continuous-time systems.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Linear state space layer (S4D)](../src/models/ssm) `linear-ssm` | reduced | full | — | [A. Gu et al. 2022](https://arxiv.org/abs/2111.00396) |
| [Mamba (selective state space model)](../src/models/ssm) `mamba` | reduced | full | — | [A. Gu, T. Dao 2023](https://arxiv.org/abs/2312.00752) |

**Linear state space layer (S4D).** A continuous-time linear system with diagonal state matrix, discretised by zero-order hold. The same layer runs as a recurrence or as one long convolution with kernel K. *Scale:* One single-input single-output S4D layer with N complex modes (S4D-Lin initialisation, zero-order hold), trained alone on a filtering task. The published models stack many such layers with H channels each, nonlinearities, normalisation and position-wise mixing, and compute the kernel with FFT convolution.

- Diagonal A only (S4D). The DPLR (normal plus low-rank) HiPPO matrix of S4 is not implemented.
- The convolution is a dense causal Toeplitz product, O(L²), not the FFT convolution used in practice; the recurrence is a sequential loop.
- A single linear layer without nonlinearity, so it can only represent linear time-invariant filters. Its kernel is a sum of N damped complex exponentials, so a pure delay is approximated, not matched.
- B is trainable here; the minimal S4D code folds it into C.

**Mamba (selective state space model).** A diagonal state space model whose step size Δ and matrices B and C are computed from each token, so the recurrence can choose what to store and what to ignore. *Scale:* Mamba blocks as in the reference implementation (RMSNorm, in-projection to 2·d, causal depthwise convolution of width 4, SiLU, input-dependent Δ, B, C, A = −exp(A_log), D skip, SiLU gate, out-projection, residual, tied embedding) at d_model = 16, expansion 2, state size N = 8, Δ rank ⌈d/16⌉ = 1 and 2 layers. The paper’s language models use d_model 768–2560, N = 16 and 24–64 layers.

- The selective scan is a sequential loop in JavaScript. The hardware-aware parallel scan with kernel fusion and recomputation, which is what makes Mamba fast on GPUs, is not reproduced; this changes speed, not the result.
- One synthetic task (selective copying) with sequences of tens of tokens instead of thousands.
- B̄ = Δ B (the Euler simplification used in the reference code) rather than the full zero-order hold for B; Ā uses zero-order hold.
- The rescaling of out-projection weights by 1/√layers at initialisation follows the reference code; weight decay and other training details of the paper are not used.

## Neural dynamics

Hidden states defined by learned differential equations.

| Model | Scale | Training here | Pretrained weights | Reference |
|---|---|---|---|---|
| [Neural ODE](../src/models/neural-dynamics) `neural-ode` | reduced | full | — | [R. T. Q. Chen et al. 2018](https://arxiv.org/abs/1806.07366) |
| [Latent ODE](../src/models/neural-dynamics) `latent-ode` | reduced | full | — | [R. T. Q. Chen et al. 2018](https://arxiv.org/abs/1806.07366) |

**Neural ODE.** The hidden state is the solution of an ordinary differential equation whose right-hand side is a neural network; points are classified after flowing along it. *Scale:* The continuous-depth model and its use as a classifier follow Chen et al. (2018). Reduced: fixed-step Euler, midpoint or RK4 instead of the adaptive Dormand–Prince solver, gradients by backpropagation through the solver steps instead of the adjoint ODE, and two-dimensional data (optionally augmented with zero dimensions as in Dupont et al., 2019) instead of MNIST.

- Discretise-then-optimise: gradients are exact derivatives of the discretised solver, obtained by backpropagating through every stage. Memory grows linearly with the number of steps. The continuous adjoint method (constant memory) is not implemented.
- Fixed step size; there is no error control, so a coarse step can give a flow that differs from the true ODE solution.
- With zero augmentation the exact flow is a homeomorphism of the plane, so classes that enclose each other (concentric circles) cannot be separated by the final linear classifier; a coarse fixed-step solver is not injective in general, so the discretised map may separate them where the exact flow cannot.
- With augmentation the state has more than two dimensions. The vector field is then drawn on the plane where the extra coordinates are zero and trajectories are projected onto the first two coordinates.

**Latent ODE.** A recognition network reads an observed trajectory and infers an initial latent state; an ODE in latent space carries it forward, and a decoder maps the latent path back to observations, also beyond the observed window. *Scale:* The generative model and variational objective follow the latent ODE of Chen et al. (2018, §5): an RNN run backwards over the observations gives q(z₀), the latent path is an ODE solution, and the decoder is applied at the observation times; the four-dimensional latent space is the paper’s. Reduced: the recognition RNN is a GRU (the ODE-RNN encoder of Rubanova et al. 2019 is not implemented), observations are regularly sampled, and the solvers are fixed-step with backpropagation through the steps instead of adaptive with the adjoint.

- Observations are regularly sampled (every 0.15 s); the irregular sampling that motivates Rubanova et al. is not exercised.
- The encoder is a GRU, not an ODE-RNN, so the recognition network has no continuous-time dynamics between observations.
- Gradients come from backpropagating through the fixed-step solver; the adjoint method is not implemented.
- The observation noise σ in the likelihood is a fixed hyperparameter, not learned.
- With a latent dimension above two (the default is four), the latent phase portrait shows the first two coordinates, with the field evaluated where the remaining coordinates are zero; paths can therefore cross in the picture. Set the latent dimension to 2 for an exact portrait; whether that size trains within a few hundred steps depends on the seed.
- Gradient checks use central differences with ε = 1e-4 instead of 1e-6: with σ = 0.05 the likelihood scales squared errors by 1/(2σ²) = 200, and float64 round-off in L(θ ± ε) then dominates the difference quotient at ε = 1e-6.
- The latent space is not identified: any invertible reparameterisation of z gives the same likelihood, so the latent coordinates need not resemble (θ, θ′).
