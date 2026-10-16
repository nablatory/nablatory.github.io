import type { GradCheckResult, OptimizerConfig, ValueEdit } from '../../engine';
import type { Config, ConfigValue, ProcedureResult } from '../../models/types';
import { engine } from '../../runtime/client';
import type { Inspected, VjpResult } from '../../runtime/protocol';
import type { GradTarget, SessionSummary, StepDetail, TrainRecord } from '../../runtime/session';
import type { Trace, TensorPayload } from '../../runtime/trace';
import type { CheckpointFile } from '../../runtime/checkpoint';
import { createStore, type Store } from '../../state/store';
import type { SavedExperiment } from '../../persist/experiments';

export type Selection = { kind: 'node'; id: number } | { kind: 'module'; name: string } | null;

export interface ExperimentState {
  status: 'opening' | 'ready' | 'error';
  busy: string | null;
  progress: number | null;
  error: string | null;
  summary: SessionSummary | null;
  trace: Trace | null;
  /** Trace before the most recent recomputation, for differences. */
  previous: Trace | null;
  /** What caused the most recent recomputation. */
  cause: string | null;
  selection: Selection;
  history: TrainRecord[];
  stepDetail: StepDetail | null;
  /**
   * Procedure results with the revision and step they were computed at; a
   * result is out of date when `rev` differs from the current revision.
   */
  procedures: Record<
    string,
    {
      result: ProcedureResult;
      ms: number;
      args: Config;
      rev: number;
      step: number;
      /** Paused before the end: the result covers only part of the run. */
      stopped: boolean;
    }
  >;
  /** Incremented whenever parameters, configuration or input change (every new inspection). */
  rev: number;
  gradTarget: GradTarget;
  gradcheck: GradCheckResult | null;
  provenance: CheckpointFile['provenance'] | null;
  vjp: { key: string; result: VjpResult | null } | null;
  /** Training in visible chunks (the Guide): steps done of the total requested. */
  live: { done: number; total: number } | null;
}

let counter = 0;

/** Owns one worker session and mirrors its state for the interface. */
export class Experiment {
  readonly id = `s${++counter}`;
  readonly store: Store<ExperimentState>;
  private closed = false;
  private halt = false;

  constructor(readonly modelId: string) {
    this.store = createStore<ExperimentState>({
      status: 'opening',
      busy: 'Opening',
      progress: null,
      error: null,
      summary: null,
      trace: null,
      previous: null,
      cause: null,
      selection: null,
      history: [],
      stepDetail: null,
      procedures: {},
      gradTarget: { kind: 'loss' },
      gradcheck: null,
      provenance: null,
      vjp: null,
      live: null,
      rev: 0,
    });
  }

  get state(): ExperimentState {
    return this.store.get();
  }

  private async run<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    this.store.set({ busy: label, progress: null, error: null });
    try {
      const r = await fn();
      if (!this.closed) this.store.set({ busy: null, progress: null });
      return r;
    } catch (e) {
      if (!this.closed)
        this.store.set({
          busy: null,
          progress: null,
          error: e instanceof Error ? e.message : String(e),
        });
      return undefined;
    }
  }

  /** Install a new inspection; keep the old one for differences unless the architecture changed. */
  private apply(r: Inspected, cause: string, sameArchitecture = true, bump = true) {
    const prev = this.state.trace;
    const keep = sameArchitecture && prev && prev.nodes.length === r.trace.nodes.length;
    const sel = this.state.selection;
    const selection =
      sel?.kind === 'node' && !r.trace.nodes.some((n) => n.id === sel.id) ? null : sel;
    this.store.set({
      summary: r.summary,
      trace: r.trace,
      previous: keep ? prev : null,
      cause,
      selection,
      vjp: null,
      rev: bump ? this.state.rev + 1 : this.state.rev,
    });
  }

  async open(opts: { config?: Config; seed?: number; checkpoint?: string | null } = {}) {
    // React may close and reopen the same experiment (StrictMode, remounts).
    this.closed = false;
    await this.run('Opening', async () => {
      const r = await engine.request(
        { type: 'open', session: this.id, model: this.modelId, ...opts },
        (p) => p.type === 'load' && this.store.set({ busy: p.label }),
      );
      this.store.set({
        status: 'ready',
        history: [],
        stepDetail: null,
        procedures: {},
        gradcheck: null,
        provenance: (r.provenance as CheckpointFile['provenance'] | undefined) ?? null,
        selection: null,
      });
      this.apply(r, 'Opened', false);
    });
    if (this.state.error) this.store.set({ status: 'error' });
  }

  close() {
    this.closed = true;
    this.halt = true;
    // Stop a running loop first: the worker runs one session's commands in order.
    engine.stop(this.id);
    engine.request({ type: 'close', session: this.id }).catch(() => undefined);
  }

  async inspect(target: GradTarget = this.state.gradTarget) {
    this.store.set({ gradTarget: target });
    await this.run('Computing', async () => {
      const r = await engine.request({ type: 'inspect', session: this.id, target });
      this.apply(
        r,
        target.kind === 'loss'
          ? 'Gradient of the loss'
          : target.kind === 'output'
            ? `Gradient of ${target.name}[${target.index}]`
            : 'Forward only',
        true,
        false,
      );
    });
  }

  async setConfig(patch: Record<string, ConfigValue>) {
    await this.run('Recomputing', async () => {
      const r = await engine.request({ type: 'setConfig', session: this.id, patch });
      const what = Object.entries(patch)
        .map(([k, v]) => `${k} = ${v}`)
        .join(', ');
      if (r.reinitialized) this.store.set({ history: [], stepDetail: null, provenance: null });
      this.apply(
        r,
        r.reinitialized
          ? `Architecture changed (${what}); parameters re-initialised`
          : `Changed ${what}`,
        !r.reinitialized,
      );
    });
  }

  async setExample(index: number) {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'setExample', session: this.id, index }),
        `Input: example ${index}`,
      );
    });
  }

  async setCustomInput(input: TensorPayload) {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'setCustomInput', session: this.id, input }),
        'Input: your image',
      );
    });
  }

  async editParam(param: string, index: number, value: number) {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'editParam', session: this.id, param, index, value }),
        `Edited ${param}[${index}]`,
      );
    });
  }

  async revertEdits() {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'revertEdits', session: this.id }),
        'Reverted parameter edits',
      );
    });
  }

  async setOverride(key: string, edits: ValueEdit[]) {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'setOverride', session: this.id, key, edits }),
        edits.length ? `Patched ${edits.length} value(s) of ${key}` : `Removed patch on ${key}`,
      );
    });
  }

  async setOptimizer(optimizer: Partial<OptimizerConfig>, batchSize?: number) {
    const prevKind = this.state.summary?.optimizer.kind;
    await this.run('Updating', async () => {
      const r = await engine.request({
        type: 'setOptimizer',
        session: this.id,
        optimizer,
        batchSize,
        resetState: optimizer.kind !== undefined && optimizer.kind !== prevKind,
      });
      this.store.set({ summary: r.summary });
    });
  }

  async setFrozen(names: string[]) {
    await this.run('Recomputing', async () => {
      this.apply(
        await engine.request({ type: 'setFrozen', session: this.id, names }),
        'Changed frozen parameters',
      );
    });
  }

  async reinit(seed: number) {
    await this.run('Initialising', async () => {
      const r = await engine.request({ type: 'reinit', session: this.id, seed });
      this.store.set({ history: [], stepDetail: null, provenance: null });
      this.apply(r, `Re-initialised with seed ${seed}`);
    });
  }

  async loadCheckpoint(checkpoint: string) {
    await this.run('Loading weights', async () => {
      const r = await engine.request({ type: 'loadCheckpoint', session: this.id, checkpoint });
      this.store.set({
        history: [],
        stepDetail: null,
        provenance: r.provenance as CheckpointFile['provenance'],
      });
      this.apply(r, `Loaded checkpoint ${checkpoint}`);
    });
  }

  async step() {
    await this.run('One optimisation step', async () => {
      const r = await engine.request({ type: 'step', session: this.id });
      this.store.set((s) => ({ history: [...s.history, r.detail.record], stepDetail: r.detail }));
      this.apply(r, `Optimisation step ${r.detail.record.step}`);
    });
  }

  async train(steps: number, label = `Training ${steps} steps`) {
    await this.run(label, async () => {
      const r = await engine.request({ type: 'train', session: this.id, steps }, (p) => {
        if (p.type !== 'train') return;
        this.store.set((s) => ({ history: [...s.history, p.record], progress: p.done / p.total }));
      });
      this.store.set({ history: r.history, stepDetail: null });
      this.apply(
        r,
        r.stopped
          ? `Training paused at step ${r.summary.step}`
          : `Trained to step ${r.summary.step}`,
      );
    });
  }

  /**
   * Train in chunks so the interface can redraw between them; `between` runs
   * after every chunk (e.g. re-running a procedure whose result is on screen).
   */
  async trainLive(steps: number, between?: () => Promise<void>) {
    this.halt = false;
    await this.liveLoop(steps, between);
  }

  private async liveLoop(steps: number, between?: () => Promise<void>) {
    const chunk = Math.max(1, Math.min(50, Math.round(steps / 15)));
    let done = 0;
    this.store.set({ live: { done, total: steps } });
    while (done < steps && !this.halt && !this.closed) {
      const n = Math.min(chunk, steps - done);
      await this.train(n, `Training: step ${done + 1}–${done + n} of ${steps}`);
      if (this.state.error) break;
      done += n;
      if (this.closed) return;
      this.store.set({ live: { done, total: steps } });
      if (between && !this.halt) await between();
    }
    if (!this.closed) this.store.set({ live: null });
  }

  /**
   * Back to the state the page opened in, apply a configuration patch, then
   * optionally train in visible chunks, as one action that Pause can interrupt.
   */
  async replay(patch: Config, steps = 0, between?: () => Promise<void>) {
    this.halt = false;
    this.store.set({ live: { done: 0, total: steps } });
    const go = () => !this.halt && !this.closed && !this.state.error;
    await this.open();
    if (go()) await this.setConfig(patch);
    if (go() && steps > 0) await this.liveLoop(steps, between);
    if (!this.closed) this.store.set({ live: null });
  }

  stop() {
    this.halt = true;
    engine.stop(this.id);
  }

  async procedure(procedure: string, args: Config, label = 'Running') {
    await this.run(label, async () => {
      const r = await engine.request(
        { type: 'procedure', session: this.id, procedure, args },
        (p) => {
          if (p.type === 'procedure') this.store.set({ progress: p.fraction });
        },
      );
      this.store.set((s) => ({
        procedures: {
          ...s.procedures,
          [procedure]: {
            result: r.result,
            ms: r.ms,
            args,
            rev: s.rev,
            step: s.summary?.step ?? 0,
            stopped: r.stopped,
          },
        },
      }));
    });
  }

  async gradcheck() {
    await this.run('Checking gradients', async () => {
      this.store.set({ gradcheck: await engine.request({ type: 'gradcheck', session: this.id }) });
    });
  }

  async vjp(key: string) {
    const result = await engine.request({ type: 'vjp', session: this.id, key }).catch(() => null);
    this.store.set({ vjp: { key, result } });
  }

  /** Reproducible description of the current state, optionally with parameter values. */
  async snapshot(title: string, includeWeights: boolean): Promise<SavedExperiment | null> {
    const s = this.state.summary;
    if (!s) return null;
    const weights = includeWeights
      ? await engine.request({ type: 'exportParams', session: this.id })
      : undefined;
    return {
      format: 'nablatory-experiment',
      version: 1,
      id: crypto.randomUUID(),
      title,
      created: new Date().toISOString(),
      model: s.modelId,
      config: s.config,
      seed: s.seed,
      checkpoint: s.checkpoint,
      optimizer: s.optimizer,
      batchSize: s.batchSize,
      frozen: s.params.filter((p) => p.frozen).map((p) => p.name),
      edits: s.edits.map(({ param, index, value }) => ({ param, index, value })),
      overrides: s.overrides,
      example: s.example === 'custom' ? 0 : s.example,
      step: s.step,
      history: this.state.history,
      weights,
    };
  }

  /** Rebuild a saved experiment: configuration, initial weights, then edits or saved weights. */
  async restore(e: SavedExperiment) {
    await this.run('Restoring', async () => {
      await engine.request({
        type: 'open',
        session: this.id,
        model: e.model,
        config: e.config,
        seed: e.seed,
        checkpoint: e.checkpoint,
      });
      await engine.request({
        type: 'setOptimizer',
        session: this.id,
        optimizer: e.optimizer,
        batchSize: e.batchSize,
        resetState: true,
      });
      await engine.request({ type: 'setFrozen', session: this.id, names: e.frozen });
      if (e.weights) {
        await engine.request({
          type: 'loadParams',
          session: this.id,
          manifest: e.weights.manifest,
          data: e.weights.data.slice(),
          note: `Restored weights saved after ${e.step} step(s)`,
        });
      }
      if (!e.weights)
        for (const ed of e.edits)
          await engine.request({ type: 'editParam', session: this.id, ...ed });
      for (const o of e.overrides)
        await engine.request({ type: 'setOverride', session: this.id, key: o.key, edits: o.edits });
      const r = await engine.request({
        type: 'setExample',
        session: this.id,
        index: typeof e.example === 'number' ? e.example : 0,
      });
      this.store.set({
        status: 'ready',
        history: e.history ?? [],
        stepDetail: null,
        provenance: null,
      });
      this.apply(r, `Restored “${e.title}”`, false);
    });
  }

  select(selection: Selection) {
    this.store.set({ selection, vjp: null });
  }
}
