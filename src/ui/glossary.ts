/**
 * Short explanations shown when the pointer rests on a term: what it does, and why it is there.
 * Each entry is written to be read in a few seconds; the equations live in the inspector.
 */
export interface GlossaryEntry {
  term: string;
  what: string;
  why?: string;
  /** Matched against setting labels, metric names and prose (case-insensitive). */
  match?: RegExp;
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // Blocks of the computation graph.
  input: {
    term: 'Input',
    what: 'The data this pass starts from, stored as a tensor.',
    why: 'Everything below it in the graph is computed from these numbers and the weights.',
  },
  source: {
    term: 'Given tensor',
    what: 'A tensor that enters the computation from outside the graph, such as drawn noise or a fixed target.',
    why: 'Nothing is learned here and no gradient flows into it; it is held fixed while the loss is differentiated.',
  },
  affine: {
    term: 'Affine (fully connected) layer',
    what: 'Multiplies the input by a learned weight matrix and adds a bias: y = Wx + b.',
    why: 'Every output can mix every input. With a nonlinearity in between, stacks of these can bend the input space into almost any shape.',
    match: /^(fully connected|dense|affine)\b/i,
  },
  conv: {
    term: 'Convolution',
    what: 'Slides small learned filters over the image and takes a weighted sum of each neighbourhood.',
    why: 'The same filter is reused at every position, so an edge or stroke is detected wherever it appears, with far fewer weights than a dense layer.',
    match: /\bconv(olution)?s?\b/i,
  },
  deconv: {
    term: 'Transposed convolution',
    what: 'Spreads every input value over a small output patch through a learned filter, enlarging the feature map.',
    why: 'The way back up in resolution: from a small code or feature map to a full image.',
    match: /up-?convolution|transposed conv/i,
  },
  attention: {
    term: 'Attention',
    what: 'Each position forms a query, compares it with every position’s key, and takes a softmax-weighted average of their values.',
    why: 'Any position can read from any other in one step, with weights that depend on the content rather than on distance.',
    match: /\battention\b/i,
  },
  embedding: {
    term: 'Embedding',
    what: 'Looks up a learned vector for each token, patch position or other index.',
    why: 'Turns discrete symbols into numbers the network can compute with; symbols used alike end up with similar vectors.',
    match: /\bembeddings?\b/i,
  },
  norm: {
    term: 'Normalisation',
    what: 'Shifts and rescales values to mean 0 and variance 1, then applies a learned scale and offset.',
    why: 'Keeps activations in a steady range while the weights change, so training is faster and less sensitive to the learning rate.',
    match: /\b(batch ?norm|layer ?norm|normali[sz]ation|RMS ?norm)\b/i,
  },
  pool: {
    term: 'Pooling',
    what: 'Replaces each small window by one value: its maximum or its mean.',
    why: 'Halves the size of the feature map and makes the result less sensitive to small shifts of the input.',
    match: /\bpool(ing)?\b|^subsampling$/i,
  },
  upsample: {
    term: 'Upsampling',
    what: 'Repeats each value to enlarge the feature map.',
    why: 'Restores resolution in a decoder without adding weights.',
  },
  subsample: {
    term: 'Subsampling shortcut',
    what: 'Keeps every second pixel and pads the missing channels with zeros.',
    why: 'Gives the shortcut the same shape as the block output when the block halves the resolution, without adding weights.',
  },
  recurrent: {
    term: 'Recurrent step',
    what: 'Combines the current input with the state carried over from the previous time step.',
    why: 'The state is the network’s memory: it lets the output at time t depend on everything seen before.',
  },
  ode: {
    term: 'ODE solver',
    what: 'Integrates dz/dt = f(z) forward in time with fixed numerical steps, where f is a small network.',
    why: 'Depth becomes continuous time: each solver step acts like a layer, and all steps share the weights of f.',
  },
  sample: {
    term: 'Random draw',
    what: 'Combines fresh random numbers with its input: added noise, a corruption mask, or μ + σ·ε.',
    why: 'Makes the computation stochastic. Writing the noise as a separate input keeps the result differentiable with respect to the weights.',
  },
  softmax: {
    term: 'Softmax',
    what: 'Turns a vector of scores into probabilities: exponentiate each score, then divide by the total.',
    why: 'Gives a distribution over classes or positions; a slightly larger score gets exponentially more weight.',
    match: /\bsoftmax\b/i,
  },
  loss: {
    term: 'Loss',
    what: 'One number that measures how far the outputs are from what they should be.',
    why: 'Training changes the weights to make it smaller; its gradient says in which direction.',
    match: /^loss$|\bloss\b/i,
  },
  kl: {
    term: 'KL divergence',
    what: 'How different one probability distribution is from another; zero only when they are equal.',
    why: 'Pulls the learned distribution towards a simple prior, which keeps the latent space smooth enough to sample from.',
    match: /\bKL\b/,
  },
  add: {
    term: 'Addition',
    what: 'Adds tensors entry by entry.',
    why: 'In a residual connection the block’s input is added back to its output, so the block only learns a correction and gradients get a direct path back.',
  },
  mul: {
    term: 'Elementwise product',
    what: 'Multiplies tensors entry by entry.',
    why: 'Used as a gate: a factor between 0 and 1 decides how much of another signal passes.',
  },
  concat: {
    term: 'Concatenation',
    what: 'Places tensors side by side along one axis.',
    why: 'Lets the next layer read several sources at once, such as a skip connection next to an upsampled path.',
  },
  activation: {
    term: 'Activation function',
    what: 'A fixed nonlinear function (ReLU, tanh, sigmoid …) applied to every entry.',
    why: 'Without it, a stack of linear layers would collapse into a single linear map.',
    match: /\bactivation\b/i,
  },
  reshape: {
    term: 'Rearrangement',
    what: 'Reshapes, slices or reorders the same numbers.',
    why: 'Only the indexing changes; no new value is computed.',
  },
  ssm: {
    term: 'Selective scan',
    what: 'Runs the linear recurrence h_t = Ā_t h_{t−1} + B̄_t x_t, with step size and matrices computed from the input.',
    why: 'Input-dependent parameters let the layer decide what to keep and what to forget, at a cost linear in the sequence length.',
  },
  graphconv: {
    term: 'Graph convolution',
    what: 'Each node combines its neighbours’ features with its own, then applies a shared weight matrix.',
    why: 'Information travels one edge per layer, so each node is classified from its surroundings.',
  },
  stopgrad: {
    term: 'Stop-gradient',
    what: 'Passes a value on but blocks the gradient from flowing back through it.',
    why: 'Lets one part of the loss treat another part’s output as a fixed target.',
  },
  returns: {
    term: 'Return',
    what: 'The discounted sum of future rewards along a trajectory.',
    why: 'This is what the agent tries to make large.',
    match: /\breturn\b/i,
  },
  targets: {
    term: 'Targets',
    what: 'The values a prediction is compared with in the loss.',
    why: 'Computed without gradients, so learning moves the prediction towards the target and not the other way round.',
  },
  value: {
    term: 'Value (Q) head',
    what: 'Predicts the discounted return still to come from a state and action.',
    why: 'Lets a short plan account for what happens after its last step.',
  },
  discretize: {
    term: 'Discretisation',
    what: 'Turns the continuous-time parameters (A, B and the step Δ) into the matrices of a per-step recurrence.',
    why: 'The layer is defined in continuous time; Δ sets how quickly it reacts to its input.',
  },
  kernel: {
    term: 'SSM kernel',
    what: 'The impulse response of the recurrence, K_k = C Ā^k B̄.',
    why: 'Convolving the input with K gives the same output as running the recurrence step by step.',
  },
  mean: { term: 'Mean', what: 'Averages over one or more axes.' },
  op: { term: 'Operation', what: 'A step of the forward computation recorded in the graph.' },

  preact: {
    term: 'Pre-activation block',
    what: 'Normalisation and ReLU come before each convolution instead of after it.',
    why: 'The shortcut then carries the block input unchanged all the way through the network.',
    match: /pre-activation/i,
  },
  l2norm: {
    term: 'L2 normalisation',
    what: 'Divides each hidden vector by its length, so it lies on the unit sphere.',
    why: 'Keeps the scale of node embeddings the same from layer to layer.',
    match: /L2-normali/i,
  },
  timing: {
    term: 'Forward and backward time',
    what: 'Wall-clock time of the last forward pass and of its backward pass, measured on this device.',
    why: 'Measured, so it varies from run to run and between devices.',
    match: /forward, backward/i,
  },

  // Activations and other single operations.
  relu: {
    term: 'ReLU',
    what: 'max(0, x): negative values become 0, positive values pass unchanged.',
    why: 'Cheap and does not saturate for positive inputs, so gradients pass through active units undiminished.',
    match: /\bReLU\b/i,
  },
  tanh: {
    term: 'tanh',
    what: 'Squashes every value into (−1, 1).',
    why: 'Bounded and centred on zero, which keeps recurrent states from growing without limit.',
    match: /\btanh\b/i,
  },
  sigmoid: {
    term: 'Sigmoid',
    what: 'Squashes every value into (0, 1).',
    why: 'Reads as a probability or a gate that is open, closed or in between.',
    match: /\bsigmoid\b/i,
  },
  gelu: {
    term: 'GELU',
    what: 'x times the probability that a standard normal variable is below x; a smooth version of ReLU.',
    why: 'The activation used in BERT and GPT.',
    match: /\bGELU\b/i,
  },
  silu: {
    term: 'SiLU',
    what: 'x · sigmoid(x), a smooth ReLU-like curve.',
    match: /\b(SiLU|swish)\b/i,
  },
  elu: {
    term: 'ELU',
    what: 'x for x > 0, eˣ − 1 below: like ReLU, but smooth and negative for x < 0.',
  },
  leakyRelu: {
    term: 'Leaky ReLU',
    what: 'x for x > 0, a small slope times x below.',
    why: 'Units never go completely silent, so a gradient always flows.',
    match: /leaky ?relu/i,
  },
  matmul: {
    term: 'Matrix product',
    what: 'Multiplies two matrices: each output entry is a dot product of a row and a column.',
  },
  maskedFill: {
    term: 'Mask',
    what: 'Replaces chosen entries (here, attention scores) by −∞ before the softmax.',
    why: 'Those positions then get exactly zero weight.',
  },
  dropout: {
    term: 'Dropout',
    what: 'During training, sets a random fraction of values to zero and scales the rest up to compensate. Off during evaluation.',
    why: 'Units cannot rely on particular partners, which reduces overfitting.',
    match: /\bdropout\b|P_drop/i,
  },
  crossEntropy: {
    term: 'Cross-entropy',
    what: 'Minus the log of the probability given to the correct class.',
    why: 'Small when the right answer gets high probability, and very large when the model is confidently wrong.',
    match: /cross-?entropy/i,
  },
  mse: {
    term: 'Mean squared error',
    what: 'The average of the squared differences between prediction and target.',
    why: 'Penalises large errors much more than small ones.',
    match: /\bMSE\b|squared error/i,
  },

  // Training.
  gradient: {
    term: 'Gradient ∂L/∂x',
    what: 'For every entry of x, how much the loss would change per unit change of that entry.',
    why: 'Its sign and size say which way, and how strongly, to move x to lower the loss.',
    match: /\bgradients?\b|∂L/i,
  },
  backward: {
    term: 'Backward pass',
    what: 'Applies the chain rule from the loss back through every recorded operation, giving the gradient of every tensor.',
    why: 'One backward pass costs about as much as one or two forward passes and yields all gradients at once.',
    match: /backward|backprop/i,
  },
  forward: {
    term: 'Forward pass',
    what: 'Runs the network from the inputs to the loss and records every operation.',
    why: 'The recorded graph is what the backward pass walks in reverse.',
    match: /\bforward\b/i,
  },
  parameters: {
    term: 'Parameters',
    what: 'The numbers training adjusts: weight matrices, filters, biases, scales.',
    why: 'Everything the model learns is stored in them.',
    match: /^parameters$|\bweights\b/i,
  },
  flops: {
    term: 'FLOPs',
    what: 'Floating-point operations of one forward pass, counted from the tensor shapes (a multiply–add counts as 2).',
    why: 'A measure of the amount of arithmetic that does not depend on the device.',
    match: /FLOPs?( per forward pass)?/,
  },
  step: {
    term: 'Training step',
    what: 'Draw a batch, compute the loss, backpropagate, update the weights.',
    why: 'Training is many of these; each one moves the weights a little downhill on that batch’s loss.',
    match: /^step$/i,
  },
  batch: {
    term: 'Batch size',
    what: 'How many examples are averaged into one gradient estimate.',
    why: 'Larger batches give smoother, more expensive steps; smaller ones are noisier and cheaper.',
    match: /batch size/i,
  },
  lr: {
    term: 'Learning rate η',
    what: 'Scales every weight update.',
    why: 'Too small and training crawls; too large and the loss jumps around or diverges.',
    match: /learning rate/i,
  },
  optimiser: {
    term: 'Optimiser',
    what: 'The rule that turns gradients into weight updates: plain SGD, momentum, RMSProp, Adam …',
    why: 'Rules differ in how fast they move along steep and along flat directions.',
    match: /^optimi[sz]er$/i,
  },
  momentum: {
    term: 'Momentum μ',
    what: 'Moves along a running average of past gradients instead of the latest one alone.',
    why: 'Noise averages out and consistent directions build up speed.',
    match: /momentum/i,
  },
  beta1: {
    term: 'β₁',
    what: 'How slowly Adam’s running average of gradients forgets old values.',
    why: 'Higher means smoother, slower-reacting steps.',
    match: /^β₁$/,
  },
  beta2: {
    term: 'β₂ / smoothing α',
    what: 'How slowly the running average of squared gradients forgets old values.',
    why: 'Each weight’s step is divided by the root of this average, so weights with large gradients take smaller steps.',
    match: /^β₂$|smoothing α/i,
  },
  weightDecay: {
    term: 'Weight decay',
    what: 'Shrinks every weight slightly towards zero at each step.',
    why: 'Discourages large weights, which often generalise worse.',
    match: /weight decay/i,
  },
  clip: {
    term: 'Gradient clipping',
    what: 'Caps the gradient before the update, by its overall length or entry by entry.',
    why: 'One huge gradient cannot throw the weights far away.',
    match: /clipping|max ‖g‖|max \|g\|/i,
  },
  seed: {
    term: 'Seed',
    what: 'The starting number of the random generator.',
    why: 'The same seed gives the same initial weights, batches and noise, so a run can be repeated exactly.',
    match: /^seed$/i,
  },
  init: {
    term: 'Initialisation',
    what: 'How the starting weights are drawn before training.',
    why: 'The scale matters: too large and signals explode through the layers, too small and they fade.',
    match: /initiali[sz]ation/i,
  },
  checkpoint: {
    term: 'Trained weights',
    what: 'Parameters saved after a longer training run in this same engine.',
    why: 'Lets you inspect a network that already works without waiting for it to train.',
    match: /trained weights|checkpoint/i,
  },
  freeze: {
    term: 'Trainable layers',
    what: 'A switched-off layer is frozen: its gradients are still computed, but its weights are not updated.',
    why: 'Shows how much of the result depends on that layer learning.',
    match: /trainable layers|frozen/i,
  },
  validation: {
    term: 'Validation',
    what: 'Loss and metrics on held-out examples that training never uses.',
    why: 'Tells real learning apart from memorising the training batches.',
    match: /\bvalidation\b/i,
  },
  accuracy: {
    term: 'Accuracy',
    what: 'The fraction of examples whose most probable class is the correct one.',
    match: /accuracy/i,
  },

  // Checks.
  gradcheck: {
    term: 'Gradient check',
    what: 'Compares each backpropagated gradient entry with (L(θ + ε) − L(θ − ε)) / 2ε, computed in float64.',
    why: 'If the two agree, the backward pass computes the true derivative of this loss.',
    match: /gradient check/i,
  },
  relError: {
    term: 'Relative error',
    what: '|analytic − numeric| divided by the larger of their sizes.',
    why: 'Below about 10⁻⁵ the two agree up to rounding error.',
    match: /relative error|max rel/i,
  },
  kink: {
    term: 'Kink',
    what: 'A point where the loss is not differentiable, such as ReLU at 0 or a max that changes winner.',
    why: 'A finite difference that straddles a kink disagrees with the analytic gradient although nothing is wrong.',
    match: /\bkinks?\b/i,
  },
  float64: {
    term: 'float64',
    what: 'Double-precision numbers, about 16 significant digits.',
    why: 'Finite differences subtract two nearly equal losses, so they need far more precision than training does.',
    match: /float64|double precision/i,
  },

  // Tensors and structure.
  shape: {
    term: 'Shape',
    what: 'The size of each axis of a tensor, for example batch × channels × height × width.',
    match: /^shape$/i,
  },
  tensor: {
    term: 'Tensor',
    what: 'An array of numbers with any number of axes.',
    match: /\btensors?\b/i,
  },
  skip: {
    term: 'Skip connection',
    what: 'Carries a block’s input past it and adds or concatenates it to the output.',
    why: 'Gradients get a short path back to early layers, and the block only has to learn a correction.',
    match: /skip connections?|shortcuts?|residual/i,
  },
  heads: {
    term: 'Attention heads',
    what: 'Several attention operations side by side, each in a smaller subspace of the model width.',
    why: 'Different heads can follow different relations at the same time.',
    match: /\bheads?\b/i,
  },
  width: {
    term: 'Width',
    what: 'The number of units or channels in a layer.',
    why: 'More width means more capacity and more computation.',
    match: /\bwidth\b|hidden units|hidden size|channels/i,
  },
  depth: {
    term: 'Depth',
    what: 'The number of stacked layers or blocks.',
    why: 'Each layer composes the features of the one before, so depth builds more abstract features.',
    match: /\b(layers|blocks|stages)\b|blocks per stage|residual blocks/i,
  },
  causal: {
    term: 'Causal mask',
    what: 'Stops each position from attending to positions after it.',
    why: 'Needed when predicting the next token; without it the model could read the answer.',
    match: /(attention )?mask\b/i,
  },
  layerNormPlacement: {
    term: 'Pre-LN and post-LN',
    what: 'Post-LN normalises after adding the sublayer to its input; pre-LN normalises the sublayer’s input.',
    why: 'Pre-LN keeps an unnormalised path through the whole stack, which tends to train more easily.',
    match: /layernorm placement/i,
  },
  sequence: {
    term: 'Sequence length',
    what: 'How many time steps or tokens one example has.',
    match: /sequence length|context length|span length/i,
  },
  patch: {
    term: 'Patch size',
    what: 'The side of the square image pieces that become tokens.',
    why: 'Smaller patches mean more tokens: finer detail, more computation.',
    match: /patch size/i,
  },

  // Latent-variable and generative models.
  latent: {
    term: 'Latent code',
    what: 'The small vector the network compresses its input into.',
    why: 'Forcing information through a narrow code makes the network keep what matters most.',
    match: /\blatent\b|code size/i,
  },
  elbo: {
    term: '−ELBO',
    what: 'Reconstruction error plus the KL term: an upper bound on −log p(x).',
    why: 'Lowering it raises a guaranteed lower bound on how likely the model finds the data.',
    match: /ELBO/i,
  },
  klWeight: {
    term: 'KL weight β',
    what: 'Multiplies the KL term of the loss.',
    why: 'Larger β trades sharp reconstructions for a smoother, more prior-like latent space.',
    match: /KL (weight|scale)/i,
  },
  noise: {
    term: 'Noise level σ',
    what: 'The standard deviation of the random noise added to the data.',
    match: /noise σ|noise$|point noise|pixel noise/i,
  },
  diffusionSteps: {
    term: 'Diffusion steps T',
    what: 'How many small noising steps turn data into pure noise.',
    why: 'Sampling reverses them one by one, so more steps mean finer, slower generation.',
    match: /diffusion steps|sampling steps/i,
  },
  schedule: {
    term: 'Noise schedule',
    what: 'How much noise βₜ each diffusion step adds.',
    why: 'Sets how quickly the signal is destroyed, and so how hard each denoising step is.',
    match: /noise schedule/i,
  },
  ddimEta: {
    term: 'η (stochasticity)',
    what: 'How much fresh noise each DDIM sampling step adds.',
    why: 'η = 0 makes sampling deterministic: the same starting noise always gives the same sample.',
    match: /^η/,
  },
  modes: {
    term: 'Modes covered',
    what: 'How many of the target clusters have generated points near them.',
    why: 'A generator can fool the discriminator while ignoring some clusters (mode collapse); this counts it.',
    match: /modes covered/i,
  },
  dx: {
    term: 'D(x), D(G(z))',
    what: 'The discriminator’s mean probability that real points, and generated points, are real.',
    why: 'Both near 0.5 means the discriminator cannot tell them apart.',
    match: /^D\(/,
  },
  generatorLoss: {
    term: 'Generator loss',
    what: 'Minimax: minimise log(1 − D(G(z))). Non-saturating: maximise log D(G(z)).',
    why: 'Same goal; the non-saturating form still gives a strong gradient when D rejects every fake.',
    match: /generator loss/i,
  },

  // World models and control.
  freeNats: {
    term: 'Free nats',
    what: 'KL below this amount per time step is not penalised.',
    why: 'Stops the KL term from flattening the latent state before the model has learned to use it.',
    match: /free nats/i,
  },
  discount: {
    term: 'Discount γ',
    what: 'A reward k steps ahead counts γᵏ times as much as one now.',
    why: 'Values near 1 look far ahead; smaller values focus on what happens soon.',
    match: /discount/i,
  },
  horizon: {
    term: 'Horizon',
    what: 'How many steps ahead the model imagines or plans.',
    why: 'Longer horizons see further but compound the model’s errors.',
    match: /horizon/i,
  },
  lambda: {
    term: 'λ-return',
    what: 'Mixes returns that bootstrap from the value estimate after 1, 2, … steps, with weights (1 − λ)λⁿ.',
    why: 'λ = 0 trusts the critic after one step; λ = 1 sums the imagined rewards to the horizon.',
    match: /λ \(return/i,
  },
  stochasticState: {
    term: 'Deterministic and stochastic state',
    what: 'h is carried forward exactly by a GRU; z is drawn each step from a learned Gaussian.',
    why: 'h remembers over many steps, z expresses what cannot be predicted.',
    match: /deterministic state|stochastic state/i,
  },
  mppi: {
    term: 'MPPI',
    what: 'Samples many action sequences, scores each with the learned model, and averages the best ones, weighted by score.',
    why: 'Plans without needing gradients of the model.',
    match: /\bMPPI\b/,
  },

  // Neural ODEs and state space models.
  solver: {
    term: 'Solver',
    what: 'The numerical method that steps the ODE forward: Euler uses one slope per step, RK4 combines four.',
    why: 'A more accurate method needs fewer steps for the same error, but evaluates f more often per step.',
    match: /\bsolver\b/i,
  },
  nfe: {
    term: 'NFE',
    what: 'Number of evaluations of f made by the solver in one pass.',
    match: /\bNFE\b/,
  },
  augmented: {
    term: 'Augmented dimensions',
    what: 'Extra zero coordinates appended to the input before the flow.',
    why: 'Flows cannot make paths cross; extra dimensions give them room to go around each other.',
    match: /augmented/i,
  },
  stepDelta: {
    term: 'Step Δ',
    what: 'The time step used to discretise the continuous-time state space model.',
    why: 'Small Δ: slow, long memory. Large Δ: quick reaction, short memory.',
    match: /step Δ|initial step/i,
  },
  stateSize: {
    term: 'State size N',
    what: 'The number of hidden state channels per input channel.',
    match: /state size|complex modes/i,
  },
  selection: {
    term: 'Selection',
    what: 'Makes Δ, B and C functions of the current input.',
    why: 'The layer can then choose, token by token, what to store and what to ignore.',
    match: /^selection \(/i,
  },

  // Recurrent networks.
  forgetBias: {
    term: 'Forget-gate bias',
    what: 'A constant added to the forget gate’s bias at initialisation.',
    why: 'A positive value starts the cell remembering by default, which helps gradients survive over long sequences.',
    match: /forget-gate|update-gate/i,
  },
  hidden: {
    term: 'Hidden state',
    what: 'The vector a recurrent network carries from one time step to the next.',
    match: /hidden state/i,
  },

  // Graph networks.
  labelled: {
    term: 'Labelled nodes',
    what: 'The few nodes whose class is used in the loss; the rest are predicted.',
    why: 'Semi-supervised: the graph lets labels spread to unlabelled neighbours.',
    match: /labelled/i,
  },
  aggregator: {
    term: 'Aggregator',
    what: 'How a node combines its neighbours’ features: mean, max of a transformed copy, and so on.',
    match: /aggregator/i,
  },
  selfLoops: {
    term: 'Self-loops',
    what: 'Adds each node to its own neighbourhood.',
    why: 'A node’s own features then count in its update.',
    match: /self-loops/i,
  },

  // Settings of particular models.
  padding: {
    term: 'Padding',
    what: 'Zeros added around the border of the input before a convolution.',
    why: 'Without it every convolution shrinks the map, and border pixels are seen less often than central ones.',
    match: /\bpadding\b/i,
  },
  stride: {
    term: 'Stride',
    what: 'How many pixels the filter moves between positions.',
    why: 'A stride of 2 halves the output size: a cheap way to shrink the feature map.',
    match: /\bstride\b/i,
  },
  c3Table: {
    term: 'C3 connection table',
    what: 'Which of the six S2 maps each of the sixteen C3 maps reads.',
    why: 'LeCun et al. connected each C3 map to a subset only, to cut connections and make the maps differ from each other.',
    match: /C3 connections/i,
  },
  outputLayer: {
    term: 'Output layer',
    what: 'Linear + softmax scores each class by a weighted sum. Euclidean RBF scores it by the squared distance to a fixed code for that class.',
    why: 'The RBF output is the paper’s; later networks use the linear one.',
    match: /^output layer$/i,
  },
  corruption: {
    term: 'Corruption',
    what: 'Damages the input before the encoder sees it: masking sets a fraction ν of the pixels to 0, salt and pepper sets them to 0 or 1, Gaussian adds noise.',
    why: 'The loss compares with the clean input, so the network has to learn what the data look like to undo the damage.',
    match: /^corruption$|corrupted fraction/i,
  },
  noiseDim: {
    term: 'Noise dimension',
    what: 'Length of the random vector z the generator turns into a sample.',
    why: 'Each z gives one sample; the generator learns to map this simple distribution onto the data.',
    match: /noise dimension/i,
  },
  lossWeight: {
    term: 'Loss coefficient',
    what: 'How much one term counts in the total loss.',
    why: 'The terms have different scales; the coefficients decide which of them the gradient follows most.',
    match: /coefficient c/i,
  },
  temporalWeight: {
    term: 'Temporal weight ρ',
    what: 'The loss at step t of the predicted rollout is multiplied by ρᵗ.',
    why: 'Predictions further ahead are less reliable, so their errors count less.',
    match: /temporal weight/i,
  },
  imagineStarts: {
    term: 'Imagination start states',
    what: 'Posterior states from the batch that the imagined trajectories start from.',
    why: 'Starting from states the model has actually seen keeps the imagined situations realistic and varied.',
    match: /imagination start/i,
  },
  integrationTime: {
    term: 'Integration time T',
    what: 'The input flows along the learned vector field from time 0 to T.',
    why: 'A longer time lets the flow carry points further.',
    match: /integration time/i,
  },
  timeInput: {
    term: 'f depends on t',
    what: 'Gives the vector field the current time as an extra input.',
    why: 'Without it the field is the same at every moment; with it the field can change along the way.',
    match: /depends on t\b/i,
  },
  likelihoodSigma: {
    term: 'Likelihood σ',
    what: 'The observation noise the decoder assumes: squared errors are divided by 2σ².',
    why: 'A smaller σ makes reconstruction errors weigh more against the KL term.',
    match: /likelihood σ/i,
  },
  reverseVariance: {
    term: 'Reverse variance σₜ²',
    what: 'The variance of the fresh noise added at each sampling step.',
    why: 'The network predicts only the noise; the variance of each reverse step is fixed by hand, as in the paper.',
    match: /reverse variance/i,
  },
  selectionRate: {
    term: 'Selection rate',
    what: 'The fraction of positions chosen for prediction in each sequence.',
    why: 'Only chosen positions enter the loss; the others are the context they are predicted from.',
    match: /selection rate|selected per sequence/i,
  },
  sampledNeighbours: {
    term: 'Sampled neighbours',
    what: 'How many neighbours, drawn at random, each node aggregates from in a layer.',
    why: 'Bounds the cost per node however many neighbours it has; 0 uses them all.',
    match: /sampled neighbours/i,
  },
  copyTokens: {
    term: 'Tokens to copy',
    what: 'How many data tokens, scattered among filler, the model has to reproduce at the end of the sequence.',
    why: 'More tokens to remember make the task harder for a fixed state size.',
    match: /tokens to copy/i,
  },
  targetFilter: {
    term: 'Target filter',
    what: 'The impulse response the model is trained to reproduce: a resonator, a delay or a moving average.',
    why: 'A known answer: the learned kernel can be compared with it directly.',
    match: /target filter/i,
  },

  // Loss terms and task measures.
  reconstruction: {
    term: 'Reconstruction error',
    what: 'How far the model’s reconstruction of its input is from the input itself.',
    why: 'It forces the code to keep the information needed to rebuild the data.',
    match: /^reconstruction$/i,
  },
  rewardLoss: {
    term: 'Reward loss',
    what: 'Error of the predicted reward against the reward in the data.',
    why: 'Planning and imagined training rely on the predicted reward.',
    match: /^reward$/i,
  },
  valueLoss: {
    term: 'Value loss',
    what: 'Error of the predicted value against its target: the reward plus the discounted value of the next state.',
    why: 'Teaches the model what happens beyond the end of a short plan.',
    match: /^value$/i,
  },
  consistency: {
    term: 'Latent consistency',
    what: 'Distance between the predicted next latent state and the encoding of the true next state.',
    why: 'Trains the dynamics without reconstructing observations: the model predicts only what the encoder keeps.',
    match: /^consistency$/i,
  },
  iou: {
    term: 'IoU',
    what: 'Intersection over union: the overlap of predicted and true regions divided by their union.',
    why: 'Unlike pixel accuracy, it is not inflated by a large, easy background.',
    match: /\bIoU\b/,
  },
  bpc: {
    term: 'Bits per character',
    what: 'The cross-entropy in bits: how many yes/no questions the model needs per character, on average.',
    why: '0 is perfect prediction; log₂ of the vocabulary size is uniform guessing.',
    match: /bits per character/i,
  },
  meanT: {
    term: 'Mean t',
    what: 'The average diffusion step drawn for the examples of this batch.',
    why: 'Each example trains at a random noise level; this shows which levels the batch covered.',
    match: /^mean t$/i,
  },
  kernelError: {
    term: 'Relative kernel error',
    what: '‖K − K*‖ / ‖K*‖ between the learned convolution kernel and the target filter.',
    why: 'Says directly whether the model has learned the filter: 0 is exact.',
    match: /kernel error/i,
  },

  // Pages.
  graph: {
    term: 'Computation graph',
    what: 'Every operation of the last forward pass, grouped into blocks; each line carries a tensor from one block to the next.',
    why: 'The backward pass walks this graph in reverse to compute every gradient.',
    match: /computation graph/i,
  },
  fidelity: {
    term: 'Scale',
    what: 'How this version compares with the published model.',
    why: 'Browser-sized versions keep the published equations with fewer layers, channels or tokens.',
  },
  changed: {
    term: 'What changed',
    what: 'The largest change in each block’s output since the previous computation.',
    why: 'Shows where a step or a setting acted.',
    match: /what changed/i,
  },
  gradientFlow: {
    term: 'Gradient flow',
    what: 'The size of the gradient at each parameter group during training.',
    why: 'Vanishing or exploding gradients show up here as layers whose values fall towards zero or blow up.',
    match: /gradient flow/i,
  },
};

const ORDERED = Object.entries(GLOSSARY).filter(([, e]) => e.match);

/** The glossary entry for an operation name in the trace. */
export const OP_TERMS: Record<string, string> = {
  linear: 'affine',
  conv2d: 'conv',
  convTranspose2d: 'deconv',
  layerNorm: 'norm',
  batchNorm: 'norm',
  rmsNorm: 'norm',
  maxPool2d: 'pool',
  avgPool2d: 'pool',
  upsampleNearest2d: 'upsample',
  subsample2d: 'subsample',
  gather: 'embedding',
  softmax: 'softmax',
  logSoftmax: 'softmax',
  relu: 'relu',
  tanh: 'tanh',
  sigmoid: 'sigmoid',
  geluTanh: 'gelu',
  silu: 'silu',
  elu: 'elu',
  leakyRelu: 'leakyRelu',
  matmul: 'matmul',
  maskedFill: 'maskedFill',
  dropout: 'dropout',
  crossEntropy: 'crossEntropy',
  mse: 'mse',
  add: 'add',
  sub: 'add',
  mul: 'mul',
  concat: 'concat',
  reshape: 'reshape',
  transpose: 'reshape',
  slice: 'reshape',
  selectiveScan: 'ssm',
  mean: 'mean',
  input: 'input',
};

export function opTerm(op: string): string | undefined {
  return OP_TERMS[op] ?? (GLOSSARY[op] ? op : undefined);
}

/** The entry whose pattern matches the longest part of a label (a setting, a metric, a heading). */
export function findTerm(label: string): string | undefined {
  let best: string | undefined;
  let len = 0;
  for (const [id, e] of ORDERED) {
    const m = e.match!.exec(label);
    if (m && m[0].length > len) {
      best = id;
      len = m[0].length;
    }
  }
  return best;
}

/** Phrases picked out in running text (guides and notes); a longer phrase comes before its parts. */
export const PROSE: [string, RegExp][] = [
  ['skip', /\b(skip connections?|residual connections?|shortcuts?)\b/i],
  ['backward', /\b(backpropagat\w*|backward pass)\b/i],
  ['crossEntropy', /\bcross-entropy\b/i],
  ['attention', /\b(self-|cross-)?attention\b/i],
  ['softmax', /\bsoftmax\b/i],
  ['deconv', /\b(transposed convolutions?|up-convolutions?)\b/i],
  ['conv', /\bconvolutions?\b/i],
  ['embedding', /\bembeddings?\b/i],
  ['norm', /\b(batch normali[sz]ation|layer normali[sz]ation|LayerNorm|batch norm)\b/i],
  ['dropout', /\bdropout\b/i],
  ['kl', /\bKL( divergence| term)?\b/],
  ['elbo', /−?ELBO\b/],
  ['lr', /\blearning rate\b/i],
  ['gradient', /\bgradients?\b/i],
  ['latent', /\blatent (space|code|state)s?\b/i],
  ['causal', /\bcausal mask\b/i],
  ['relu', /\bReLU\b/],
  ['pool', /\b(max-?pool(ing)?|average pool(ing)?)\b/i],
  ['validation', /\bvalidation (loss|set|digits)\b/i],
  ['mppi', /\bMPPI\b/],
  ['freeNats', /\bfree nats\b/i],
  ['solver', /\b(RK4|Euler) (solver|steps?)\b/i],
  ['stopgrad', /\bstop-gradient\b/i],
];

export interface ProsePart {
  text: string;
  term?: string;
}

/**
 * Splits text into plain parts and glossary terms: the first occurrence of each term, at most
 * `limit` terms, never inside a “quoted” name.
 */
export function annotate(text: string, limit = 4): ProsePart[] {
  const hits: { start: number; end: number; term: string }[] = [];
  const quoted: [number, number][] = [];
  for (const m of text.matchAll(/“[^”]*”/g)) quoted.push([m.index!, m.index! + m[0].length]);
  const free = (a: number, b: number) =>
    !quoted.some(([s, e]) => a < e && b > s) && !hits.some((h) => a < h.end && b > h.start);
  for (const [term, re] of PROSE) {
    if (hits.length >= limit) break;
    for (const m of text.matchAll(new RegExp(re.source, `${re.flags}g`))) {
      const start = m.index!;
      const end = start + m[0].length;
      if (free(start, end)) {
        hits.push({ start, end, term });
        break;
      }
    }
  }
  hits.sort((a, b) => a.start - b.start);
  const parts: ProsePart[] = [];
  let at = 0;
  for (const h of hits) {
    if (h.start > at) parts.push({ text: text.slice(at, h.start) });
    parts.push({ text: text.slice(h.start, h.end), term: h.term });
    at = h.end;
  }
  if (at < text.length) parts.push({ text: text.slice(at) });
  return parts;
}
