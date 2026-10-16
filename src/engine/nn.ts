import { Tensor, alloc } from './tensor';
import { activeGraph, graphRng, isTraining, op, type Var } from './graph';
import type { ParamBuilder } from './params';
import { linear } from './ops/linalg';
import { conv2d, type Conv2dOptions } from './ops/conv';
import { add, exp, mul, mulScalar, square, sub, addScalar } from './ops/elementwise';
import { sum, mean } from './ops/reduce';

/** Parameter of the active graph. */
export function p(name: string): Var {
  const g = activeGraph();
  if (!g) throw new Error(`parameter ${name} requested outside a graph`);
  return g.param(name);
}

export function hasParam(name: string): boolean {
  return activeGraph()?.params?.has(name) ?? false;
}

/** y = x Wᵀ + b using parameters `${name}.weight` and `${name}.bias` (if present). */
export function dense(x: Var, name: string): Var {
  return linear(x, p(`${name}.weight`), hasParam(`${name}.bias`) ? p(`${name}.bias`) : undefined);
}

export function conv(x: Var, name: string, opts: Conv2dOptions = {}): Var {
  return conv2d(
    x,
    p(`${name}.weight`),
    hasParam(`${name}.bias`) ? p(`${name}.bias`) : undefined,
    opts,
  );
}

/** Inverted dropout: in training, zero with probability `rate` and scale survivors by 1/(1−rate). */
export function dropout(x: Var, rate: number): Var {
  if (!isTraining() || rate <= 0) return x;
  const rng = graphRng();
  const keep = 1 - rate;
  const m = alloc(x.value.size, x.value.data);
  for (let i = 0; i < m.length; i++) m[i] = rng.uniform() < keep ? 1 / keep : 0;
  const mask = new Tensor(x.shape, m);
  const out = alloc(m.length, m);
  for (let i = 0; i < m.length; i++) out[i] = x.value.data[i] * m[i];
  return op(
    'dropout',
    new Tensor(x.shape, out),
    [x],
    (g) => {
      const gx = alloc(m.length, m);
      for (let i = 0; i < m.length; i++) gx[i] = g.data[i] * m[i];
      return [new Tensor(x.shape, gx)];
    },
    { rate },
    { mask },
  );
}

/** KL(N(μ, e^{logσ²}) ‖ N(0, I)), summed over the last axis and averaged over rows. */
export function klStandardNormal(mu: Var, logvar: Var): Var {
  // ½ Σ (μ² + σ² − log σ² − 1)
  const t = sub(add(square(mu), exp(logvar)), addScalar(logvar, 1));
  return mulScalar(mean(sum(t, -1)), 0.5);
}

/** KL(N(μq, σq²) ‖ N(μp, σp²)) for diagonal Gaussians given log-std, summed over the last axis, averaged over rows. */
export function klDiagGaussians(muQ: Var, logStdQ: Var, muP: Var, logStdP: Var): Var {
  // log σp − log σq + (σq² + (μq − μp)²) / (2σp²) − ½
  const varQ = exp(mulScalar(logStdQ, 2));
  const num = add(varQ, square(sub(muQ, muP)));
  const ratio = mul(num, exp(mulScalar(logStdP, -2)));
  const t = addScalar(add(sub(logStdP, logStdQ), mulScalar(ratio, 0.5)), -0.5);
  return mean(sum(t, -1));
}

// Parameter declaration helpers using PyTorch default initialisation
// (weights and biases ~ U(±1/√fan_in)) unless stated otherwise.

export function declareLinear(
  b: ParamBuilder,
  name: string,
  nIn: number,
  nOut: number,
  opts: { bias?: boolean; init?: 'torch' | 'he' | 'xavier' | 'zeros'; std?: number } = {},
): void {
  const init = opts.init ?? 'torch';
  if (opts.std !== undefined) {
    b.add(`${name}.weight`, [nOut, nIn], { kind: 'normal', std: opts.std });
  } else if (init === 'he') {
    b.add(`${name}.weight`, [nOut, nIn], { kind: 'heNormal', fanIn: nIn });
  } else if (init === 'xavier') {
    b.add(`${name}.weight`, [nOut, nIn], { kind: 'xavierUniform', fanIn: nIn, fanOut: nOut });
  } else if (init === 'zeros') {
    b.add(`${name}.weight`, [nOut, nIn], { kind: 'zeros' });
  } else {
    b.add(`${name}.weight`, [nOut, nIn], { kind: 'lecunUniform', fanIn: nIn });
  }
  if (opts.bias ?? true) {
    b.add(
      `${name}.bias`,
      [nOut],
      init === 'torch' && opts.std === undefined
        ? { kind: 'lecunUniform', fanIn: nIn }
        : { kind: 'zeros' },
    );
  }
}

export function declareConv(
  b: ParamBuilder,
  name: string,
  cIn: number,
  cOut: number,
  k: number,
  opts: {
    bias?: boolean;
    groups?: number;
    init?: 'torch' | 'he' | 'normal';
    std?: number;
    kw?: number;
  } = {},
): void {
  const groups = opts.groups ?? 1;
  const kw = opts.kw ?? k;
  const fanIn = (cIn / groups) * k * kw;
  const shape = [cOut, cIn / groups, k, kw];
  if (opts.init === 'he') b.add(`${name}.weight`, shape, { kind: 'heNormal', fanIn });
  else if (opts.init === 'normal')
    b.add(`${name}.weight`, shape, { kind: 'normal', std: opts.std ?? 0.02 });
  else b.add(`${name}.weight`, shape, { kind: 'lecunUniform', fanIn });
  if (opts.bias ?? true) {
    b.add(
      `${name}.bias`,
      [cOut],
      opts.init === undefined || opts.init === 'torch'
        ? { kind: 'lecunUniform', fanIn }
        : { kind: 'zeros' },
    );
  }
}

export function declareNorm(b: ParamBuilder, name: string, dim: number): void {
  b.add(`${name}.weight`, [dim], { kind: 'ones' });
  b.add(`${name}.bias`, [dim], { kind: 'zeros' });
}
