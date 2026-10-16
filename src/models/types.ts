import type { Tensor, Var, ParamStore, Rng, OptimizerConfig } from '../engine';

export type FamilyId =
  | 'feedforward'
  | 'convolutional'
  | 'recurrent'
  | 'transformer'
  | 'autoencoder'
  | 'diffusion'
  | 'graph'
  | 'gan'
  | 'world-model'
  | 'ssm'
  | 'neural-dynamics';

export interface Reference {
  authors: string;
  title: string;
  year: number;
  url?: string;
}

/**
 * exact     – the published architecture at its published size.
 * equations – the published equations of a model whose paper fixes no size
 *             (perceptron, Elman network, LSTM, GRU …); the size here is a choice.
 * reduced   – the defining equations of the published architecture, with fewer
 *             layers / channels / tokens so it runs and trains in a browser.
 */
export type Fidelity = 'exact' | 'equations' | 'reduced';

/** Plain-language guide shown at the basic level of a model page. */
export interface Guide {
  /** Two or three sentences on what the model does, for a first-time reader. */
  what: string;
  /** What to look at while it trains. */
  watch: string;
  /** Suggested changes, applied through the configuration; `train` runs that many steps after applying. */
  try?: { label: string; note: string; patch: Config; train?: number }[];
  /** Titles of the views to show large (default: the first two). */
  views?: string[];
}

export type ConfigValue = number | string | boolean;
export type Config = Record<string, ConfigValue>;

/**
 * architecture – changes parameter shapes; parameters are re-initialised.
 * forward      – changes the computation without changing parameter shapes.
 * data         – changes the dataset or input.
 */
export type ConfigEffect = 'architecture' | 'forward' | 'data';

interface FieldBase {
  key: string;
  label: string;
  affects: ConfigEffect;
  help?: string;
}

export type ConfigField =
  | (FieldBase & { kind: 'int' | 'float'; min: number; max: number; step?: number; log?: boolean })
  | (FieldBase & { kind: 'select'; options: { value: string; label: string }[] })
  | (FieldBase & { kind: 'bool' })
  | (FieldBase & {
      kind: 'text';
      maxLength?: number;
      /** Suggested values, offered as a list under the input. */
      suggestions?: string[];
      /** Returns a message explaining why the text cannot be used, or null. */
      validate?: (s: string) => string | null;
    });

/** A batch of examples. Leading axis of every input tensor is the batch axis. */
export interface Batch {
  inputs: Record<string, Tensor>;
  targets?: Record<string, Tensor>;
  /** Integer class targets, one per example. */
  labels?: number[];
  meta?: BatchMeta;
}

export interface BatchMeta {
  /** Display strings for the tokens of the first sequence. Other string-array keys may label attention rows/columns. */
  tokens?: string[];
  /** Display names for class indices. */
  classNames?: string[];
  /** Short description of the example(s). */
  caption?: string;
  /** Graph structure for graph models: undirected edge list. */
  edges?: [number, number][];
  /** Arbitrary additional JSON-safe values for views. */
  [key: string]: unknown;
}

/** Fetches static assets (relative to the site root in the browser, to `public/` in Node). */
export type AssetLoader = (path: string) => Promise<ArrayBuffer>;

export interface Dataset<C extends Config = Config> {
  id: string;
  name: string;
  description: string;
  /** Where the data comes from. */
  source: 'synthetic' | 'embedded' | 'static';
  license?: string;
  /** Load static assets; must be called before sampling when `source === 'static'`. */
  load?(loader: AssetLoader): Promise<void>;
  /** Random training batch. */
  train(rng: Rng, batchSize: number, config: C): Batch;
  /** Fixed held-out batch, if the dataset has one. */
  validation?(config: C): Batch;
  /** Number of fixed inspection examples. */
  exampleCount: number;
  example(index: number, config: C): Batch;
  exampleLabel(index: number, config: C): string;
}

export interface ForwardResult {
  /** Named tensors referenced by views (logits, probabilities, reconstructions …). */
  outputs: Record<string, Var>;
  /** Scalar training objective, when the batch has targets. */
  loss?: Var;
  /** Scalar diagnostics computed from the outputs (accuracy, KL term …). */
  metrics?: Record<string, number>;
}

export interface ForwardContext {
  training: boolean;
  rng: Rng;
}

export interface TrainPhase<C extends Config = Config> {
  id: string;
  label: string;
  /** Parameters updated in this phase; all others are frozen while it runs. */
  params: (name: string) => boolean;
  loss(
    batch: Batch,
    config: C,
    ctx: ForwardContext,
  ): { loss: Var; metrics?: Record<string, number> };
  /**
   * Semi-gradient objectives (TD targets, stop-gradient consistency targets)
   * cannot pass a finite-difference check as written: perturbing θ also moves
   * the targets, which backpropagation treats as constants. `checkLoss` is
   * called once, in a no-grad graph at the current parameters, to freeze those
   * targets; the returned function is the same objective with them held fixed,
   * which is what the analytic gradient differentiates.
   */
  checkLoss?(batch: Batch, config: C): (ctx: ForwardContext) => Var;
}

export interface TrainingSpec<C extends Config = Config> {
  /**
   * full      – the whole model is trained in the browser.
   * component – only part of the system is trainable here (stated in `note`).
   * none      – inference only.
   */
  support: 'full' | 'component' | 'none';
  note?: string;
  optimizer: OptimizerConfig;
  batchSize: number;
  steps: number;
  /** Multi-objective training (GAN, actor–critic). Default: one phase on `forward().loss`. */
  phases?: TrainPhase<C>[];
  /** Evaluate held-out data every N steps (0 = never). */
  validateEvery?: number;
  /**
   * Called after every optimisation step with the updated parameters, for
   * updates that are not gradient steps (exponential-moving-average target
   * networks). Writes go to non-trainable parameters only.
   */
  afterStep?(params: ParamStore, step: number, config: C): void;
}

export interface ProcedureContext<C extends Config = Config> {
  params: ParamStore;
  config: C;
  rng: Rng;
  args: Config;
  dataset: Dataset<C>;
  /** Cooperative cancellation for long loops. */
  shouldStop(): boolean;
  progress(fraction: number): void;
}

export interface ProcedureResult {
  tensors: Record<string, Tensor>;
  scalars?: Record<string, number>;
  text?: string;
  meta?: BatchMeta;
}

/** A named no-grad experiment beyond a single forward pass (sampling, rollouts, generation). */
export interface Procedure<C extends Config = Config> {
  id: string;
  label: string;
  description: string;
  args?: ConfigField[];
  defaults?: Config;
  run(ctx: ProcedureContext<C>): ProcedureResult;
}

/**
 * Tensor reference used by views:
 *   'out:<name>'            forward output (expose targets/labels as outputs when a view needs them)
 *   'in:<name>'             batch input
 *   'proc:<id>:<tensor>'    procedure result tensor ('proc:<id>:text' for its text)
 */
export type TensorRef = string;

/**
 * Plot extent [x0, x1, y0, y1]: fixed numbers, or a reference to a 4-element
 * tensor computed by a procedure (for data whose range is not known in advance).
 * A grid-evaluated field must have been computed on the same extent.
 */
export type Extent = [number, number, number, number] | TensorRef;

/** Fields every view accepts. */
export interface ViewCommon {
  /** Show the view only for configurations where it applies (e.g. "stage 4" only when there are 4 stages). */
  when?: (config: Config) => boolean;
  /** Shown instead of "Not computed yet." when the source tensor is absent, e.g. a view that applies to another task. */
  unavailable?: string;
}

export type ViewSpec = ViewCommon &
  (
    | { kind: 'probs'; title: string; source: TensorRef; classNames?: string[] }
    | {
        kind: 'image';
        title: string;
        source: TensorRef;
        range?: [number, number];
        caption?: string;
      }
    | { kind: 'featureMaps'; title: string; source: TensorRef }
    | {
        kind: 'imageGrid';
        title: string;
        source: TensorRef;
        range?: [number, number];
        columns?: number;
      }
    | {
        kind: 'attention';
        title: string;
        source: TensorRef;
        tokens?: 'meta';
        /** Batch-meta keys holding the query (row) and key (column) labels; both default to "tokens". */
        queryTokens?: string;
        keyTokens?: string;
      }
    | {
        kind: 'series';
        title: string;
        sources: { ref: TensorRef; label: string }[];
        xLabel?: string;
        /** x values (one per point); the index is used when omitted. */
        x?: TensorRef;
        logY?: boolean;
        note?: string;
      }
    | {
        kind: 'scatter';
        title: string;
        source: TensorRef;
        colorBy?: TensorRef;
        field?: TensorRef;
        extent?: Extent;
      }
    | {
        kind: 'graph';
        title: string;
        source: TensorRef;
        colorBy?: TensorRef | 'argmax';
        classNames?: string[];
      }
    | { kind: 'matrix'; title: string; source: TensorRef }
    | { kind: 'vector'; title: string; source: TensorRef; labels?: string[] }
    | { kind: 'tokens'; title: string; source: TensorRef; topK?: number; note?: string }
    | { kind: 'text'; title: string; source: TensorRef }
    | {
        kind: 'rollout';
        title: string;
        predicted: TensorRef;
        observed?: TensorRef;
        reward?: TensorRef;
        rewardObserved?: TensorRef;
      }
    | {
        kind: 'phase';
        title: string;
        field: TensorRef;
        trajectories?: TensorRef;
        /** Class index per trajectory, for colouring. */
        colorBy?: TensorRef;
        extent: Extent;
        note?: string;
      }
  );

export interface MathEntry {
  tex: string;
  text?: string;
}

export interface Checkpoint {
  id: string;
  label: string;
  /** Path relative to the site root, e.g. "weights/lenet5-mnist". `.json` and `.bin` are appended. */
  path: string;
  description: string;
}

/**
 * A user-supplied image. The interface delivers a [1, channels, height, width]
 * tensor with intensities in [0, 1] (ink = 1 for drawn digits);
 * `buildCustomBatch` applies the model's own preprocessing.
 */
export interface CustomInput {
  kind: 'image';
  channels: 1 | 3;
  height: number;
  width: number;
  /** Invert uploaded photos so dark ink on light paper becomes ink = 1. */
  invertUploads: boolean;
  preprocessing: string;
}

export interface ModelDef<C extends Config = Config> {
  id: string;
  family: FamilyId;
  name: string;
  summary: string;
  /** The defining equation in one line of TeX, shown in the catalog. */
  signature: string;
  fidelity: Fidelity;
  /** What was reduced relative to the published model (required when fidelity = reduced). */
  scaleNote?: string;
  references: Reference[];
  limitations: string[];
  configSchema: ConfigField[];
  defaults: C;
  /** Returns an explanation when a configuration is mathematically invalid. */
  validate?(config: C): string | null;
  init(config: C, rng: Rng): ParamStore;
  datasets: Dataset<C>[];
  forward(batch: Batch, config: C, ctx: ForwardContext): ForwardResult;
  training: TrainingSpec<C>;
  procedures?: Procedure<C>[];
  views: ViewSpec[];
  /** Equations keyed by scope prefix (e.g. "conv1"), or by its stem without trailing digits ("block" matches "block3"). */
  math: Record<string, MathEntry>;
  checkpoints?: Checkpoint[];
  /**
   * Finite-difference settings when the defaults (ε = 1e-6, tolerance 1e-5) do not suit the
   * model's loss scale. State the reason in a comment; never loosen the tolerance to hide an error.
   */
  gradCheck?: { eps?: number; tol?: number };
  customInput?: CustomInput;
  buildCustomBatch?(input: Tensor, config: C): Batch;
  /** Basic-level text. Every sentence must hold for this implementation at this scale. */
  guide?: Guide;
}

// Helper for authoring with a concrete config type while storing in a common list.
export function defineModel<C extends Config>(def: ModelDef<C>): ModelDef {
  return def as unknown as ModelDef;
}

export interface FamilyInfo {
  id: FamilyId;
  name: string;
  summary: string;
}
