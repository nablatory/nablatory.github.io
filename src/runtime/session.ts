import {
  Graph,
  Optimizer,
  Rng,
  Tensor,
  backward,
  gradCheck,
  defaultOptimizer,
  noGrad,
  type GradCheckResult,
  type OptimizerConfig,
  type ParamStore,
  type ValueEdit,
  type Var,
} from '../engine';
import type {
  AssetLoader,
  Batch,
  Config,
  ConfigValue,
  Dataset,
  ModelDef,
  ProcedureResult,
  TrainPhase,
} from '../models/types';
import { serializeGraph, type Trace, type TensorPayload } from './trace';

export interface ParamEdit {
  param: string;
  index: number;
  value: number;
  previous: number;
}

export type GradTarget =
  { kind: 'loss' } | { kind: 'output'; name: string; index: number } | { kind: 'none' };

export interface TrainRecord {
  step: number;
  /** Loss per training phase (single-phase models use "main"). */
  losses: Record<string, number>;
  metrics: Record<string, number>;
  gradNorm: number;
  clipScale: number;
  params: Record<string, { gradNorm: number; updateNorm: number; weightNorm: number }>;
  validation?: { loss: number; metrics: Record<string, number> };
  ms: number;
}

export interface Lineage {
  /** How the current parameter values came about, oldest first. */
  events: string[];
}

export interface SessionSummary {
  modelId: string;
  config: Config;
  seed: number;
  step: number;
  paramCount: number;
  trainableCount: number;
  params: {
    name: string;
    shape: number[];
    trainable: boolean;
    init: string;
    source: string;
    frozen: boolean;
  }[];
  lineage: string[];
  edits: ParamEdit[];
  overrides: { key: string; edits: ValueEdit[] }[];
  optimizer: OptimizerConfig;
  batchSize: number;
  example: number | 'custom';
  exampleLabels: string[];
  datasetId: string;
  checkpoint: string | null;
  invalid: string | null;
}

export interface StepDetail {
  record: TrainRecord;
  before: Trace;
  after: Trace;
  updates: Record<string, TensorPayload>;
}

function payload(t: Tensor): TensorPayload {
  return { shape: t.shape, data: Float32Array.from(t.data) };
}

/**
 * One configured model instance: parameters, optimiser state, inputs and
 * training history. Used by the browser worker, the tests and the offline
 * training scripts alike.
 */
export class Session {
  readonly model: ModelDef;
  config: Config;
  seed: number;
  params!: ParamStore;
  step = 0;
  history: TrainRecord[] = [];
  edits: ParamEdit[] = [];
  overrides = new Map<string, ValueEdit[]>();
  frozen = new Set<string>();
  optimizerConfig: OptimizerConfig;
  batchSize: number;
  example: number | 'custom' = 0;
  custom: Batch | null = null;
  /** The drawing or upload behind `custom`, kept so the batch can be rebuilt for a new configuration. */
  private customInput: Tensor | null = null;
  checkpoint: string | null = null;
  lineage: string[] = [];
  datasetIndex = 0;
  invalid: string | null = null;
  /** Graph of the most recent inspection, kept for per-operation VJP queries. */
  private lastGraph: Graph | null = null;
  private optimizers = new Map<string, Optimizer>();
  private trainRng: Rng;

  constructor(model: ModelDef, opts: { config?: Partial<Config>; seed?: number } = {}) {
    this.model = model;
    this.config = { ...model.defaults, ...(opts.config ?? {}) } as Config;
    this.seed = opts.seed ?? 1;
    this.optimizerConfig = { ...defaultOptimizer, ...model.training.optimizer };
    this.batchSize = model.training.batchSize;
    this.trainRng = new Rng(this.seed).fork('train');
    this.invalid = model.validate?.(this.config) ?? null;
    if (this.invalid) throw new Error(this.invalid);
    this.reinitialize();
  }

  get dataset(): Dataset {
    return this.model.datasets[this.datasetIndex];
  }

  async loadData(loader: AssetLoader): Promise<void> {
    await this.dataset.load?.(loader);
  }

  reinitialize(seed = this.seed): void {
    this.seed = seed;
    this.params = this.model.init(this.config, new Rng(seed).fork('init'));
    this.trainRng = new Rng(seed).fork('train');
    this.resetTrainingState();
    this.edits = [];
    this.checkpoint = null;
    this.lineage = [`Initialised with seed ${seed}`];
    for (const name of [...this.frozen]) if (!this.params.has(name)) this.frozen.delete(name);
  }

  private resetTrainingState(): void {
    this.step = 0;
    this.history = [];
    this.optimizers.clear();
  }

  /** Apply a configuration patch. Returns whether parameters were re-initialised. */
  setConfig(patch: Record<string, ConfigValue>): { reinitialized: boolean } {
    const next = { ...this.config, ...patch };
    const problem = this.model.validate?.(next) ?? null;
    if (problem) throw new Error(problem);
    const archChanged = this.model.configSchema.some(
      (f) => f.affects === 'architecture' && next[f.key] !== this.config[f.key],
    );
    this.config = next;
    if (archChanged) {
      this.overrides.clear();
      this.reinitialize();
      this.lineage = [`Architecture changed; re-initialised with seed ${this.seed}`];
    }
    if (this.example !== 'custom' && this.example >= this.dataset.exampleCount) this.example = 0;
    // The model's preprocessing may depend on the configuration (ViT's patch tokens).
    if (this.customInput && this.model.buildCustomBatch)
      this.custom = this.model.buildCustomBatch(this.customInput, this.config);
    return { reinitialized: archChanged };
  }

  setOptimizer(cfg: Partial<OptimizerConfig>, batchSize?: number): void {
    this.optimizerConfig = { ...this.optimizerConfig, ...cfg };
    for (const o of this.optimizers.values()) o.config = this.optimizerConfig;
    if (batchSize) this.batchSize = batchSize;
  }

  /** Changing the optimiser kind discards its moment estimates. */
  resetOptimizerState(): void {
    for (const o of this.optimizers.values()) o.reset();
  }

  editParam(name: string, index: number, value: number): void {
    const t = this.params.get(name).clone();
    if (index < 0 || index >= t.size) throw new Error(`index ${index} out of range for ${name}`);
    const previous = t.data[index];
    t.data[index] = value;
    this.params.set(name, t);
    this.params.setSource(name, 'edited');
    this.edits.push({ param: name, index, value, previous });
    this.lineage.push(
      `Edited ${name}[${index}]: ${previous.toPrecision(4)} → ${value.toPrecision(4)}`,
    );
  }

  /** Undo all parameter edits in reverse order. */
  revertEdits(): void {
    for (const e of [...this.edits].reverse()) {
      const t = this.params.get(e.param).clone();
      t.data[e.index] = e.previous;
      this.params.set(e.param, t);
    }
    if (this.edits.length) this.lineage.push(`Reverted ${this.edits.length} edit(s)`);
    this.edits = [];
  }

  setOverride(key: string, edits: ValueEdit[]): void {
    if (edits.length) this.overrides.set(key, edits);
    else this.overrides.delete(key);
  }

  loadCheckpoint(
    id: string,
    manifest: { name: string; shape: number[]; offset: number }[],
    data: Float32Array,
  ): void {
    this.params.load(manifest, data, 'pretrained');
    this.resetTrainingState();
    this.edits = [];
    this.checkpoint = id;
    this.lineage = [`Loaded checkpoint “${id}”`];
  }

  /** Replace parameter values (saved experiment); the training history restarts from here. */
  restoreParams(
    manifest: { name: string; shape: number[]; offset: number }[],
    data: Float32Array,
    note: string,
  ): void {
    this.params.load(manifest, data, 'trained');
    this.edits = [];
    this.lineage.push(note);
  }

  inspectionBatch(): Batch {
    if (this.example === 'custom' && this.custom) return this.custom;
    const i = this.example === 'custom' ? 0 : this.example;
    return this.dataset.example(i, this.config);
  }

  setCustomInput(t: Tensor): void {
    if (!this.model.buildCustomBatch) throw new Error('model does not accept custom input');
    this.custom = this.model.buildCustomBatch(t, this.config);
    this.customInput = t;
    this.example = 'custom';
  }

  private phases(): TrainPhase[] {
    if (this.model.training.phases?.length) return this.model.training.phases;
    const model = this.model;
    return [
      {
        id: 'main',
        label: 'Loss',
        params: () => true,
        loss: (batch, config, ctx) => {
          const r = model.forward(batch, config, ctx);
          if (!r.loss) throw new Error('forward pass produced no loss for a training batch');
          return { loss: r.loss, metrics: r.metrics };
        },
      },
    ];
  }

  /**
   * Forward pass on the inspection input with every node recorded, followed
   * by a backward pass from the chosen objective.
   */
  inspect(
    target: GradTarget = { kind: 'loss' },
    batch = this.inspectionBatch(),
    training = false,
  ): Trace {
    const rng = new Rng(this.seed).fork(`inspect:${this.step}`);
    const g = new Graph({
      params: this.params,
      record: true,
      grad: target.kind !== 'none',
      training,
      frozen: this.frozen,
      overrides: this.overrides,
      rng,
    });
    // An inspection never changes state: training-mode layers (batch-norm running
    // statistics) may write to non-trainable tensors, so restore them afterwards.
    const saved = training
      ? this.params
          .names()
          .filter((n) => !this.params.isTrainable(n))
          .map((n) => [n, Float64Array.from(this.params.get(n).data)] as const)
      : [];
    const t0 = performance.now();
    let res: ReturnType<ModelDef['forward']>;
    try {
      res = g.run(() => this.model.forward(batch, this.config, { training, rng }));
    } finally {
      for (const [n, d] of saved) this.params.get(n).data.set(d);
    }
    const t1 = performance.now();
    let objective: Trace['objective'];
    if (target.kind === 'loss' && res.loss?.requiresGrad) {
      backward(res.loss);
      objective = { node: res.loss.id, label: 'loss' };
    } else if (target.kind === 'output') {
      const out = res.outputs[target.name];
      if (out?.requiresGrad) {
        const seed = Tensor.zeros(out.shape, out.value.data);
        seed.data[target.index] = 1;
        backward(out, seed);
        objective = { node: out.id, label: `${target.name}[${target.index}]` };
      }
    }
    const t2 = performance.now();
    this.lastGraph = g;
    return serializeGraph(g, res.outputs, {
      objective,
      loss: res.loss?.value.data[0],
      metrics: res.metrics ?? {},
      timing: { forwardMs: t1 - t0, backwardMs: t2 - t1 },
      training,
      meta: batch.meta,
    });
  }

  /**
   * The vector–Jacobian products of one operation from the last inspection:
   * its upstream gradient ∂L/∂y and, for each input, the contribution
   * (∂y/∂x)ᵀ ∂L/∂y that this operation alone sends back.
   */
  vjp(key: string): {
    upstream: TensorPayload;
    parents: { key: string; contribution: TensorPayload | null; total: TensorPayload | null }[];
  } | null {
    const node = this.lastGraph?.find(key);
    if (!node?.grad || !node.backwardFn) return null;
    const gs = node.backwardFn(node.grad);
    return {
      upstream: payload(node.grad),
      parents: node.parents.map((p, i) => ({
        key: p.key,
        contribution: gs[i] && p.requiresGrad ? payload(gs[i]!) : null,
        total: p.grad ? payload(p.grad) : null,
      })),
    };
  }

  evaluate(): { loss: number; metrics: Record<string, number> } | null {
    const batch = this.dataset.validation?.(this.config);
    if (!batch) return null;
    const rng = new Rng(this.seed).fork('eval');
    const g = new Graph({ params: this.params, record: false, grad: false, training: false, rng });
    const res = g.run(() => this.model.forward(batch, this.config, { training: false, rng }));
    if (!res.loss) return null;
    return { loss: res.loss.value.data[0], metrics: res.metrics ?? {} };
  }

  private optimizerFor(phase: string): Optimizer {
    let o = this.optimizers.get(phase);
    if (!o) {
      o = new Optimizer(this.optimizerConfig);
      this.optimizers.set(phase, o);
    }
    return o;
  }

  /** One optimisation step (all phases) on a fresh training batch. */
  trainStep(opts: { detail?: boolean } = {}): StepDetail | TrainRecord {
    if (this.model.training.support === 'none')
      throw new Error('this model has no trainable configuration here');
    const t0 = performance.now();
    const batch = this.dataset.train(this.trainRng, this.batchSize, this.config);
    const stepRng = this.trainRng.fork(`step:${this.step}`);
    const record: TrainRecord = {
      step: this.step + 1,
      losses: {},
      metrics: {},
      gradNorm: 0,
      clipScale: 1,
      params: {},
      ms: 0,
    };
    let before: Trace | undefined;
    const updates: Record<string, TensorPayload> = {};
    let sq = 0;
    for (const phase of this.phases()) {
      const frozen = new Set(this.frozen);
      for (const n of this.params.names()) if (!phase.params(n)) frozen.add(n);
      const g = new Graph({
        params: this.params,
        record: !!opts.detail && !before,
        grad: true,
        training: true,
        frozen,
        overrides: this.overrides,
        rng: stepRng.fork(phase.id),
      });
      let outputs: Record<string, Var> = {};
      const tf = performance.now();
      const { loss, metrics } = g.run(() => {
        if (this.model.training.phases?.length)
          return phase.loss(batch, this.config, { training: true, rng: stepRng });
        const r = this.model.forward(batch, this.config, { training: true, rng: stepRng });
        outputs = r.outputs;
        if (!r.loss) throw new Error('forward pass produced no loss for a training batch');
        return { loss: r.loss, metrics: r.metrics };
      });
      const tb = performance.now();
      if (!Number.isFinite(loss.value.data[0])) {
        throw new Error(
          `loss became ${loss.value.data[0]} at step ${this.step + 1}; lower the learning rate or enable clipping`,
        );
      }
      backward(loss);
      const te = performance.now();
      const grads = new Map<string, Tensor>();
      for (const n of this.params.names()) {
        const v = g.paramVar(n);
        if (v?.grad && v.requiresGrad) grads.set(n, v.grad);
      }
      if (opts.detail && !before) {
        before = serializeGraph(g, outputs, {
          objective: { node: loss.id, label: phase.label },
          loss: loss.value.data[0],
          metrics: metrics ?? {},
          timing: { forwardMs: tb - tf, backwardMs: te - tb },
          training: true,
        });
      }
      const rep = this.optimizerFor(phase.id).step(this.params, grads);
      record.losses[phase.id] = loss.value.data[0];
      Object.assign(record.metrics, metrics ?? {});
      sq += rep.gradNorm * rep.gradNorm;
      record.clipScale = Math.min(record.clipScale, rep.clipScale);
      Object.assign(record.params, rep.params);
      if (opts.detail) for (const [k, u] of rep.updates) updates[k] = payload(u);
    }
    record.gradNorm = Math.sqrt(sq);
    this.step++;
    this.model.training.afterStep?.(this.params, this.step, this.config);
    const last = this.lineage.length - 1;
    const m = /^Trained (\d+) step/.exec(this.lineage[last] ?? '');
    if (m) this.lineage[last] = `Trained ${Number(m[1]) + 1} steps`;
    else this.lineage.push('Trained 1 step');
    const every = this.model.training.validateEvery ?? 0;
    if (every > 0 && this.step % every === 0) record.validation = this.evaluate() ?? undefined;
    record.ms = performance.now() - t0;
    this.history.push(record);
    if (opts.detail && before) {
      // Same batch, updated parameters: the effect of exactly one update.
      const after = this.inspect({ kind: 'loss' }, batch, true);
      return { record, before, after, updates };
    }
    return record;
  }

  runProcedure(
    id: string,
    args: Config,
    hooks: { shouldStop(): boolean; progress(f: number): void },
  ): ProcedureResult {
    const proc = this.model.procedures?.find((p) => p.id === id);
    if (!proc) throw new Error(`unknown procedure ${id}`);
    const rng = new Rng(this.seed).fork(`proc:${id}:${JSON.stringify(args)}`);
    const g = new Graph({ params: this.params, record: false, grad: false, training: false, rng });
    return g.run(() =>
      proc.run({
        params: this.params,
        config: this.config,
        rng,
        args: { ...(proc.defaults ?? {}), ...args },
        dataset: this.dataset,
        ...hooks,
      }),
    );
  }

  /**
   * Float64 central differences of every training phase's loss, each on the parameters
   * that phase updates. `eps` may be raised for models whose loss scale makes ε = 1e-6
   * suffer float64 round-off.
   */
  gradCheck(names?: string[], perParam = 3, eps?: number, training = false): GradCheckResult {
    const batch = this.dataset.train(
      new Rng(this.seed).fork('gradcheck'),
      Math.min(4, this.batchSize),
      this.config,
    );
    const phases = this.phases();
    const results: GradCheckResult[] = [];
    phases.forEach((phase, k) => {
      // A phase may read what the earlier phases of the same step left on the batch
      // (Dreamer's critic regresses on the actor pass's λ-returns), so the previous phase
      // runs first, without gradients, exactly as it does within a training step.
      if (k > 0) {
        const prev = phases[k - 1];
        new Graph({
          params: this.params,
          record: false,
          grad: false,
          training,
          rng: new Rng(3),
        }).run(() => prev.loss(batch, this.config, { training, rng: new Rng(3) }));
      }
      const own = this.params
        .names()
        .filter(
          (n) => this.params.isTrainable(n) && phase.params(n) && (!names || names.includes(n)),
        );
      if (!own.length) return;
      // The engine's first evaluation is the backpropagated one, at the unperturbed
      // parameters; semi-gradient targets are frozen there and reused by every
      // perturbed evaluation.
      let fixed: ((ctx: { training: boolean; rng: Rng }) => Var) | undefined;
      const lossAt = (ctx: { training: boolean; rng: Rng }): Var => {
        if (!phase.checkLoss) return phase.loss(batch, this.config, ctx).loss;
        fixed ??= noGrad(() => phase.checkLoss!(batch, this.config));
        return fixed(ctx);
      };
      results.push(
        gradCheck(
          // Every evaluation uses identically seeded generators, so dropout masks and sampled
          // noise are the same in L(θ+ε) and L(θ−ε).
          () => lossAt({ training, rng: new Rng(3) }),
          this.params,
          {
            names: own,
            perParam,
            eps: eps ?? this.model.gradCheck?.eps,
            tol: this.model.gradCheck?.tol,
            training,
          },
        ),
      );
    });
    const entries = results.flatMap((r) => r.entries);
    return {
      entries,
      maxRelError: Math.max(0, ...results.map((r) => r.maxRelError)),
      // Every phase passes (each has at least one agreeing entry and none that disagrees).
      pass: results.length > 0 && results.every((r) => r.pass),
      kinks: results.reduce((a, r) => a + r.kinks, 0),
      eps: results[0]?.eps ?? eps ?? this.model.gradCheck?.eps ?? 1e-6,
      dtype: 'float64',
    };
  }

  summary(): SessionSummary {
    const names = this.params.names();
    return {
      modelId: this.model.id,
      config: this.config,
      seed: this.seed,
      step: this.step,
      paramCount: effectiveCount(this.params),
      trainableCount: effectiveCount(
        this.params,
        (n) => this.params.isTrainable(n) && !this.frozen.has(n),
      ),
      params: names.map((n) => ({
        name: n,
        shape: this.params.get(n).shape,
        trainable: this.params.isTrainable(n),
        init: this.params.meta(n).init,
        source: this.params.meta(n).source,
        frozen: this.frozen.has(n),
      })),
      lineage: this.lineage,
      edits: this.edits,
      overrides: [...this.overrides].map(([key, edits]) => ({ key, edits })),
      optimizer: this.optimizerConfig,
      batchSize: this.batchSize,
      example: this.example,
      exampleLabels: Array.from({ length: this.dataset.exampleCount }, (_, i) =>
        this.dataset.exampleLabel(i, this.config),
      ),
      datasetId: this.dataset.id,
      checkpoint: this.checkpoint,
      invalid: this.invalid,
    };
  }
}

/**
 * Parameter count excluding entries fixed to zero by a structural mask
 * (e.g. LeNet-5's C3 connection table, stored as "<layer>.mask").
 */
export function effectiveCount(
  params: ParamStore,
  filter: (n: string) => boolean = () => true,
): number {
  let n = 0;
  for (const name of params.names()) {
    if (!filter(name) || !params.isTrainable(name)) continue;
    const maskName = name.replace(/\.weight$/, '.mask');
    if (maskName !== name && params.has(maskName)) {
      const w = params.get(name);
      const m = params.get(maskName);
      const per = w.size / m.size;
      let ones = 0;
      for (let i = 0; i < m.size; i++) if (m.data[i] !== 0) ones++;
      n += ones * per;
    } else {
      n += params.get(name).size;
    }
  }
  return n;
}
