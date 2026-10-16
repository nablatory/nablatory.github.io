import { Tensor, alloc } from './tensor';
import type { ParamStore } from './params';

export type OptimizerKind = 'sgd' | 'momentum' | 'nesterov' | 'rmsprop' | 'adam' | 'adamw';
export type ClipMode = 'none' | 'norm' | 'value';

export interface OptimizerConfig {
  kind: OptimizerKind;
  lr: number;
  /** Momentum coefficient μ (momentum / Nesterov). */
  momentum: number;
  beta1: number;
  /** β₂ for Adam/AdamW; smoothing α for RMSProp. */
  beta2: number;
  eps: number;
  /** L2 penalty added to the gradient, except AdamW where decay is decoupled. */
  weightDecay: number;
  clip: ClipMode;
  clipValue: number;
}

export const defaultOptimizer: OptimizerConfig = {
  kind: 'adam',
  lr: 1e-3,
  momentum: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0,
  clip: 'none',
  clipValue: 1,
};

export const optimizerTex: Record<OptimizerKind, string> = {
  sgd: '\\theta \\leftarrow \\theta - \\eta\\, g',
  momentum: 'b \\leftarrow \\mu b + g,\\quad \\theta \\leftarrow \\theta - \\eta\\, b',
  nesterov: 'b \\leftarrow \\mu b + g,\\quad \\theta \\leftarrow \\theta - \\eta\\,(g + \\mu b)',
  rmsprop:
    'v \\leftarrow \\alpha v + (1-\\alpha) g^2,\\quad \\theta \\leftarrow \\theta - \\eta\\, g / (\\sqrt{v} + \\epsilon)',
  adam: 'm \\leftarrow \\beta_1 m + (1-\\beta_1) g,\\; v \\leftarrow \\beta_2 v + (1-\\beta_2) g^2,\\; \\theta \\leftarrow \\theta - \\eta\\, \\hat m / (\\sqrt{\\hat v} + \\epsilon)',
  adamw:
    '\\theta \\leftarrow (1-\\eta\\lambda)\\theta - \\eta\\, \\hat m / (\\sqrt{\\hat v} + \\epsilon)',
};

export interface ParamStepRecord {
  gradNorm: number;
  updateNorm: number;
  weightNorm: number;
}

export interface StepReport {
  step: number;
  gradNorm: number;
  clipScale: number;
  params: Record<string, ParamStepRecord>;
  /** Update Δθ actually applied to each parameter. */
  updates: Map<string, Tensor>;
}

function norm(d: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

/** Optimizers follow the PyTorch update rules (dampening 0, bias-corrected Adam). */
export class Optimizer {
  t = 0;
  private state = new Map<string, { m?: Float64Array; v?: Float64Array }>();

  constructor(public config: OptimizerConfig) {}

  reset(): void {
    this.t = 0;
    this.state.clear();
  }

  step(params: ParamStore, grads: Map<string, Tensor>): StepReport {
    const c = this.config;
    this.t++;
    let total = 0;
    for (const g of grads.values()) {
      for (let i = 0; i < g.size; i++) {
        const v = g.data[i];
        total += v * v;
      }
    }
    const gradNorm = Math.sqrt(total);
    let clipScale = 1;
    if (c.clip === 'norm' && gradNorm > c.clipValue) clipScale = c.clipValue / (gradNorm + 1e-6);

    const report: StepReport = {
      step: this.t,
      gradNorm,
      clipScale,
      params: {},
      updates: new Map(),
    };
    for (const [name, g0] of grads) {
      const theta = params.get(name);
      const n = theta.size;
      const g = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let v = g0.data[i] * clipScale;
        if (c.clip === 'value') v = Math.max(-c.clipValue, Math.min(c.clipValue, v));
        if (c.kind !== 'adamw') v += c.weightDecay * theta.data[i];
        g[i] = v;
      }
      let st = this.state.get(name);
      if (!st) {
        st = {};
        this.state.set(name, st);
      }
      const delta = alloc(n, theta.data);
      switch (c.kind) {
        case 'sgd':
          for (let i = 0; i < n; i++) delta[i] = -c.lr * g[i];
          break;
        case 'momentum':
        case 'nesterov': {
          const first = !st.m;
          const b = (st.m ??= new Float64Array(n));
          for (let i = 0; i < n; i++) {
            b[i] = first ? g[i] : c.momentum * b[i] + g[i];
            const dir = c.kind === 'nesterov' ? g[i] + c.momentum * b[i] : b[i];
            delta[i] = -c.lr * dir;
          }
          break;
        }
        case 'rmsprop': {
          const v = (st.v ??= new Float64Array(n));
          for (let i = 0; i < n; i++) {
            v[i] = c.beta2 * v[i] + (1 - c.beta2) * g[i] * g[i];
            delta[i] = (-c.lr * g[i]) / (Math.sqrt(v[i]) + c.eps);
          }
          break;
        }
        case 'adam':
        case 'adamw': {
          const m = (st.m ??= new Float64Array(n));
          const v = (st.v ??= new Float64Array(n));
          const bc1 = 1 - Math.pow(c.beta1, this.t);
          const bc2 = 1 - Math.pow(c.beta2, this.t);
          for (let i = 0; i < n; i++) {
            m[i] = c.beta1 * m[i] + (1 - c.beta1) * g[i];
            v[i] = c.beta2 * v[i] + (1 - c.beta2) * g[i] * g[i];
            const mh = m[i] / bc1;
            const vh = v[i] / bc2;
            let d = (-c.lr * mh) / (Math.sqrt(vh) + c.eps);
            if (c.kind === 'adamw') d -= c.lr * c.weightDecay * theta.data[i];
            delta[i] = d;
          }
          break;
        }
      }
      const next = alloc(n, theta.data);
      for (let i = 0; i < n; i++) next[i] = theta.data[i] + delta[i];
      params.set(name, new Tensor(theta.shape, next));
      params.setSource(name, 'trained');
      report.updates.set(name, new Tensor(theta.shape, delta));
      report.params[name] = {
        gradNorm: norm(g0.data),
        updateNorm: norm(delta),
        weightNorm: norm(next),
      };
    }
    return report;
  }
}
