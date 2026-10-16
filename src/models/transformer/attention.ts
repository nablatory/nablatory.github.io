/**
 * Shared building blocks of the transformer family: multi-head scaled
 * dot-product attention, masks, the position-wise feed-forward network,
 * residual blocks with post- or pre-layer normalisation, and sinusoidal
 * position encodings. Everything is composed from engine ops, so every
 * gradient is the engine's own VJP.
 */
import {
  Tensor,
  add,
  declareLinear,
  declareNorm,
  dense,
  dropout,
  layerNorm,
  matmul,
  maskedFill,
  mulScalar,
  p,
  relu,
  reshape,
  scope,
  softmax,
  transpose,
  type ParamBuilder,
  type Var,
} from '../../engine';
import type { ConfigField, MathEntry } from '../types';
import { geluTanh } from './ops';

export type NormPlacement = 'post' | 'pre';
export type FfnActivation = 'relu' | 'gelu';
export type LinearInit = { init: 'xavier' } | { std: number };

export const NORM_OPTIONS: { value: NormPlacement; label: string }[] = [
  { value: 'post', label: 'Post-LN: LN(x + F(x))' },
  { value: 'pre', label: 'Pre-LN: x + F(LN(x))' },
];

/** Shared validation: heads must divide the model width. */
export function headsProblem(d: number, heads: number): string | null {
  if (d % heads !== 0) {
    return `The model width d = ${d} is not divisible by ${heads} heads, so the heads cannot each take d/h = ${(d / heads).toFixed(2)} dimensions.`;
  }
  return null;
}

export const widthFields = (o: {
  dMin: number;
  dMax: number;
  dStep: number;
  maxLayers: number;
  layerLabel?: string;
}): ConfigField[] => [
  {
    key: 'layers',
    label: o.layerLabel ?? 'Layers N',
    kind: 'int',
    min: 1,
    max: o.maxLayers,
    affects: 'architecture',
  },
  {
    key: 'dModel',
    label: 'Model width d',
    kind: 'int',
    min: o.dMin,
    max: o.dMax,
    step: o.dStep,
    affects: 'architecture',
  },
  {
    key: 'heads',
    label: 'Heads h',
    kind: 'int',
    min: 1,
    max: 8,
    affects: 'architecture',
    help: 'Each head attends in a d/h-dimensional subspace.',
  },
];

// ── masks ───────────────────────────────────────────────────────────────────

/** [Tq, Tk] mask with 1 where key position k lies in the future of query q (k > q + offset). */
export function causalMask(Tq: number, Tk = Tq, offset = 0): Tensor {
  const m = Tensor.zeros([Tq, Tk]);
  for (let q = 0; q < Tq; q++) for (let k = q + offset + 1; k < Tk; k++) m.data[q * Tk + k] = 1;
  return m;
}

/**
 * [B, H, Tq, Tk] mask hiding padded keys (`pad[b][k]` true), optionally
 * combined with a causal mask. maskedFill broadcasts a mask over leading
 * axes only, so a per-example mask has to cover all four axes.
 */
export function keyPaddingMask(
  pad: boolean[][],
  heads: number,
  Tq: number,
  causal?: Tensor,
): Tensor {
  const B = pad.length;
  const Tk = pad[0].length;
  const m = Tensor.zeros([B, heads, Tq, Tk]);
  for (let b = 0; b < B; b++)
    for (let h = 0; h < heads; h++)
      for (let q = 0; q < Tq; q++)
        for (let k = 0; k < Tk; k++) {
          const hidden = pad[b][k] || (causal ? causal.data[q * Tk + k] !== 0 : false);
          if (hidden) m.data[((b * heads + h) * Tq + q) * Tk + k] = 1;
        }
  return m;
}

// ── parameters ──────────────────────────────────────────────────────────────

export function declareAttention(b: ParamBuilder, name: string, d: number, init: LinearInit): void {
  for (const k of ['q', 'k', 'v', 'o']) declareLinear(b, `${name}.${k}`, d, d, init);
}

export function declareFfn(
  b: ParamBuilder,
  name: string,
  d: number,
  dff: number,
  init: LinearInit,
): void {
  declareLinear(b, `${name}.fc1`, d, dff, init);
  declareLinear(b, `${name}.fc2`, dff, d, init);
}

/** Encoder-style block: self-attention + FFN, each with its LayerNorm (ln1, ln2). */
export function declareEncoderBlock(
  b: ParamBuilder,
  name: string,
  d: number,
  dff: number,
  init: LinearInit,
): void {
  declareNorm(b, `${name}.ln1`, d);
  declareAttention(b, `${name}.attn`, d, init);
  declareNorm(b, `${name}.ln2`, d);
  declareFfn(b, `${name}.ffn`, d, dff, init);
}

/** Decoder block: masked self-attention (ln1), cross-attention (ln2), FFN (ln3). */
export function declareDecoderBlock(
  b: ParamBuilder,
  name: string,
  d: number,
  dff: number,
  init: LinearInit,
): void {
  declareNorm(b, `${name}.ln1`, d);
  declareAttention(b, `${name}.attn`, d, init);
  declareNorm(b, `${name}.ln2`, d);
  declareAttention(b, `${name}.cross`, d, init);
  declareNorm(b, `${name}.ln3`, d);
  declareFfn(b, `${name}.ffn`, d, dff, init);
}

/** Sinusoidal encodings (Vaswani et al. 2017, §3.5): PE(pos, 2i) = sin(pos/10000^{2i/d}), PE(pos, 2i+1) = cos(·). */
export function sinusoidalEncoding(maxLen: number, d: number): Tensor {
  const t = Tensor.zeros([maxLen, d]);
  for (let pos = 0; pos < maxLen; pos++)
    for (let i = 0; i < d; i++) {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
      t.data[pos * d + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  return t;
}

// ── computation ─────────────────────────────────────────────────────────────

export function norm(x: Var, name: string): Var {
  return layerNorm(x, p(`${name}.weight`), p(`${name}.bias`));
}

export interface AttentionResult {
  /** Output after the output projection W^O, [B, Tq, d]. */
  y: Var;
  /** Attention weights softmax(QKᵀ/√d_k), [B, H, Tq, Tk]; each row sums to 1. */
  probs: Var;
}

/**
 * Multi-head scaled dot-product attention (Vaswani et al. 2017, §3.2):
 * head_i = softmax(Q_i K_iᵀ/√d_k + M) V_i, output = Concat(head_1…h) W^O.
 * `mask` (1 = hidden) is either [Tq, Tk] (shared) or [B, H, Tq, Tk]; hidden
 * scores are set to −∞, so their weights are exactly zero and receive no gradient.
 */
export function multiHeadAttention(
  xq: Var,
  xkv: Var,
  name: string,
  heads: number,
  opts: { mask?: Tensor; dropout?: number } = {},
): AttentionResult {
  const [B, Tq, D] = xq.shape;
  const Tk = xkv.shape[1];
  const dk = D / heads;
  const q = transpose(reshape(dense(xq, `${name}.q`), [B, Tq, heads, dk]), [0, 2, 1, 3]); // [B,H,Tq,dk]
  const kT = transpose(reshape(dense(xkv, `${name}.k`), [B, Tk, heads, dk]), [0, 2, 3, 1]); // [B,H,dk,Tk]
  const v = transpose(reshape(dense(xkv, `${name}.v`), [B, Tk, heads, dk]), [0, 2, 1, 3]); // [B,H,Tk,dk]
  let scores = mulScalar(matmul(q, kT), 1 / Math.sqrt(dk));
  if (opts.mask) scores = maskedFill(scores, opts.mask, -Infinity);
  const probs = softmax(scores, -1);
  const mixed = matmul(dropout(probs, opts.dropout ?? 0), v); // [B,H,Tq,dk]
  const concat = reshape(transpose(mixed, [0, 2, 1, 3]), [B, Tq, D]);
  return { y: dense(concat, `${name}.o`), probs };
}

/** Position-wise feed-forward network FFN(x) = W₂ σ(W₁x + b₁) + b₂. */
export function feedForward(x: Var, name: string, act: FfnActivation): Var {
  const h = dense(x, `${name}.fc1`);
  return dense(act === 'relu' ? relu(h) : geluTanh(h), `${name}.fc2`);
}

export interface BlockOptions {
  heads: number;
  norm: NormPlacement;
  act: FfnActivation;
  dropout: number;
  mask?: Tensor;
}

/**
 * One residual sublayer. Post-LN (Vaswani et al.): LN(x + Dropout(F(x))).
 * Pre-LN (GPT-2, ViT): x + Dropout(F(LN(x))).
 */
function sublayer<T extends { y: Var }>(
  x: Var,
  block: string,
  ln: string,
  inner: string,
  placement: NormPlacement,
  rate: number,
  f: (h: Var) => T,
): { x: Var; r: T } {
  if (placement === 'post') {
    const r = scope(inner, () => f(x));
    return { x: scope(ln, () => norm(add(x, dropout(r.y, rate)), `${block}.${ln}`)), r };
  }
  const h = scope(ln, () => norm(x, `${block}.${ln}`));
  const r = scope(inner, () => f(h));
  return { x: add(x, dropout(r.y, rate)), r };
}

/** Self-attention block (encoder layer, BERT/GPT/ViT block). Scope `name` with nested `attn`, `ffn`, `ln1`, `ln2`. */
export function encoderBlock(x: Var, name: string, o: BlockOptions): { y: Var; probs: Var } {
  return scope(name, () => {
    const a = sublayer(x, name, 'ln1', 'attn', o.norm, o.dropout, (h) =>
      multiHeadAttention(h, h, `${name}.attn`, o.heads, { mask: o.mask, dropout: o.dropout }),
    );
    const f = sublayer(a.x, name, 'ln2', 'ffn', o.norm, o.dropout, (h) => ({
      y: feedForward(h, `${name}.ffn`, o.act),
    }));
    return { y: f.x, probs: a.r.probs };
  });
}

/** Decoder layer: masked self-attention, cross-attention over the encoder output `memory`, FFN. */
export function decoderBlock(
  x: Var,
  memory: Var,
  name: string,
  o: BlockOptions & { memoryMask?: Tensor },
): { y: Var; selfProbs: Var; crossProbs: Var } {
  return scope(name, () => {
    const a = sublayer(x, name, 'ln1', 'attn', o.norm, o.dropout, (h) =>
      multiHeadAttention(h, h, `${name}.attn`, o.heads, { mask: o.mask, dropout: o.dropout }),
    );
    const c = sublayer(a.x, name, 'ln2', 'cross', o.norm, o.dropout, (h) =>
      multiHeadAttention(h, memory, `${name}.cross`, o.heads, {
        mask: o.memoryMask,
        dropout: o.dropout,
      }),
    );
    const f = sublayer(c.x, name, 'ln3', 'ffn', o.norm, o.dropout, (h) => ({
      y: feedForward(h, `${name}.ffn`, o.act),
    }));
    return { y: f.x, selfProbs: a.r.probs, crossProbs: c.r.probs };
  });
}

// ── math entries ────────────────────────────────────────────────────────────

export const ATTENTION_TEX =
  '\\operatorname{MultiHead}(Q,K,V) = \\operatorname{Concat}(\\text{head}_1,\\dots,\\text{head}_h)\\,W^O,\\quad \\text{head}_i = \\operatorname{softmax}\\!\\Big(\\frac{Q W_i^Q (K W_i^K)^\\top}{\\sqrt{d_k}} + M\\Big) V W_i^V';

export function ffnTex(act: FfnActivation): string {
  return act === 'relu'
    ? '\\operatorname{FFN}(x) = \\max(0,\\, x W_1 + b_1)\\, W_2 + b_2'
    : '\\operatorname{FFN}(x) = \\operatorname{GELU}(x W_1 + b_1)\\, W_2 + b_2,\\quad \\operatorname{GELU}(u) \\approx \\tfrac12 u\\big(1 + \\tanh\\big(\\sqrt{2/\\pi}\\,(u + 0.044715u^3)\\big)\\big)';
}

export const LN_TEX =
  '\\operatorname{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta';

/**
 * Math entries for nested scopes of numbered blocks ("enc1/attn" …): the
 * inspector looks up exact nested keys, so they are listed per layer.
 */
export function nestedMath(
  stem: string,
  maxLayers: number,
  entries: Record<string, MathEntry>,
): Record<string, MathEntry> {
  const out: Record<string, MathEntry> = {};
  for (let l = 1; l <= maxLayers; l++)
    for (const [k, v] of Object.entries(entries)) out[`${stem}${l}/${k}`] = v;
  return out;
}
