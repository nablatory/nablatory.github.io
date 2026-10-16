import {
  Tensor,
  activeGraph,
  add,
  batchNorm,
  bceWithLogits,
  graphRng,
  mulScalar,
  p,
  scope,
  sigmoidFn,
  type ParamBuilder,
  type Var,
} from '../../engine';
import type { Batch, Config, ForwardContext, TrainPhase } from '../types';
import { constant } from '../util';

export type GeneratorLoss = 'nonsaturating' | 'minimax';

export const G_LOSS_OPTIONS: { value: GeneratorLoss; label: string }[] = [
  { value: 'nonsaturating', label: 'Non-saturating: −log D(G(z))' },
  { value: 'minimax', label: 'Minimax: log(1 − D(G(z)))' },
];

/** The pieces a GAN supplies; the phases and losses are shared. */
export interface GanParts<C extends Config> {
  /** Real examples of a batch as a graph leaf. */
  real(batch: Batch): Var;
  /** n noise vectors drawn from the graph's generator. */
  noise(n: number, c: C): Var;
  /** G(z), built inside scope('G'). */
  generate(z: Var, c: C): Var;
  /** Discriminator logits [N, 1], built inside the named scope. */
  discriminate(x: Var, c: C, scopeName: string): Var;
}

/** z ~ N(0, I) as an [n, d] constant, drawn from the graph's generator. */
export function gaussianNoise(n: number, d: number): Var {
  const rng = graphRng();
  const t = Tensor.zeros([n, d]);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.normal();
  return constant(t, 'z');
}

/** Mean binary cross-entropy of logits against a constant label (1 = real, 0 = generated). */
export function bceAgainst(logits: Var, label: 0 | 1): Var {
  return bceWithLogits(logits, Tensor.full(logits.shape, label));
}

/** L_D = −E log D(x) − E log(1 − D(G(z))). */
export function discriminatorLoss(dReal: Var, dFake: Var): Var {
  return add(bceAgainst(dReal, 1), bceAgainst(dFake, 0));
}

/**
 * Generator objective to minimise:
 *   non-saturating  −E log D(G(z))
 *   minimax          E log(1 − D(G(z)))  (= −BCE against label 0)
 */
export function generatorLoss(dFake: Var, kind: GeneratorLoss): Var {
  return kind === 'nonsaturating' ? bceAgainst(dFake, 1) : mulScalar(bceAgainst(dFake, 0), -1);
}

export function meanOf(t: Tensor): number {
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.data[i];
  return s / t.size;
}

/** Mean of σ(logits), computed from the values (no graph node). */
export function meanProb(logits: Var): number {
  const d = logits.value.data;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += sigmoidFn(d[i]);
  return s / d.length;
}

export const isG = (name: string) => name.startsWith('G.');
export const isD = (name: string) => name.startsWith('D.');

/**
 * Alternating training (Goodfellow et al., 2014, algorithm 1 with k = 1):
 * a discriminator step on fresh real and generated batches with G frozen,
 * then a generator step on fresh noise with D frozen.
 */
export function ganPhases<C extends Config & { gLoss: GeneratorLoss }>(
  parts: GanParts<C>,
): TrainPhase<C>[] {
  return [
    {
      id: 'discriminator',
      label: 'Discriminator loss',
      params: isD,
      loss(batch, c) {
        const x = parts.real(batch);
        const n = x.shape[0];
        const fake = parts.generate(parts.noise(n, c), c);
        const dReal = parts.discriminate(x, c, 'D_real');
        const dFake = parts.discriminate(fake, c, 'D_fake');
        const loss = scope('loss_D', () => discriminatorLoss(dReal, dFake));
        return { loss, metrics: { 'D(x)': meanProb(dReal), 'D(G(z))': meanProb(dFake) } };
      },
    },
    {
      id: 'generator',
      label: 'Generator loss',
      params: isG,
      loss(batch, c) {
        const n = parts.real(batch).shape[0];
        const fake = parts.generate(parts.noise(n, c), c);
        const dFake = parts.discriminate(fake, c, 'D_fake');
        return { loss: scope('loss_G', () => generatorLoss(dFake, c.gLoss as GeneratorLoss)) };
      },
    },
  ];
}

/** Shared forward pass for inspection and evaluation: both losses on one batch. */
export function ganForward<C extends Config & { gLoss: GeneratorLoss }>(
  parts: GanParts<C>,
  batch: Batch,
  c: C,
  _ctx: ForwardContext,
  fakeCount?: number,
) {
  const x = parts.real(batch);
  const fake = parts.generate(parts.noise(fakeCount ?? x.shape[0], c), c);
  const dReal = parts.discriminate(x, c, 'D_real');
  const dFake = parts.discriminate(fake, c, 'D_fake');
  const dLoss = scope('loss_D', () => discriminatorLoss(dReal, dFake));
  const gLoss = scope('loss_G', () => generatorLoss(dFake, c.gLoss as GeneratorLoss));
  return { x, fake, dReal, dFake, dLoss, gLoss };
}

/* ---------- batch normalisation with running statistics stored as fixed tensors ---------- */

export function declareBatchNorm(b: ParamBuilder, name: string, channels: number): void {
  b.add(`${name}.weight`, [channels], { kind: 'ones' });
  b.add(`${name}.bias`, [channels], { kind: 'zeros' });
  b.fixed(
    `${name}.runningMean`,
    Tensor.zeros([channels]),
    'Running mean (updated in training mode, used in evaluation)',
  );
  b.fixed(
    `${name}.runningVar`,
    Tensor.full([channels], 1),
    'Running variance (updated in training mode, used in evaluation)',
  );
}

/**
 * Batch normalisation: batch statistics in training mode (and the running
 * statistics, momentum 0.1, are updated in place); running statistics otherwise.
 */
export function bn(x: Var, name: string): Var {
  const store = activeGraph()?.params;
  if (!store) throw new Error('batch norm outside a graph');
  return batchNorm(x, p(`${name}.weight`), p(`${name}.bias`), {
    mean: store.get(`${name}.runningMean`),
    var: store.get(`${name}.runningVar`),
    momentum: 0.1,
  });
}
