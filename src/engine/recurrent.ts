import type { Var } from './graph';
import { linear } from './ops/linalg';
import { add, mul, sigmoid, tanh, addScalar, neg } from './ops/elementwise';
import { slice } from './ops/shape';

/** Elman RNN cell: h' = tanh(W_ih x + b_ih + W_hh h + b_hh). */
export function rnnCell(x: Var, h: Var, wih: Var, whh: Var, bih?: Var, bhh?: Var, act = tanh): Var {
  return act(add(linear(x, wih, bih), linear(h, whh, bhh)));
}

/**
 * GRU cell, PyTorch gate layout [r | z | n]:
 * r = σ(W_ir x + b_ir + W_hr h + b_hr)
 * z = σ(W_iz x + b_iz + W_hz h + b_hz)
 * n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
 * h' = (1 − z) ⊙ n + z ⊙ h
 */
export function gruCell(
  x: Var,
  h: Var,
  wih: Var,
  whh: Var,
  bih?: Var,
  bhh?: Var,
): { h: Var; r: Var; z: Var; n: Var } {
  const H = h.shape[h.value.rank - 1];
  const gi = linear(x, wih, bih);
  const gh = linear(h, whh, bhh);
  const ax = gi.value.rank - 1;
  const r = sigmoid(add(slice(gi, ax, 0, H), slice(gh, ax, 0, H)));
  const z = sigmoid(add(slice(gi, ax, H, 2 * H), slice(gh, ax, H, 2 * H)));
  const n = tanh(add(slice(gi, ax, 2 * H, 3 * H), mul(r, slice(gh, ax, 2 * H, 3 * H))));
  const oneMinusZ = addScalar(neg(z), 1);
  const hNext = add(mul(oneMinusZ, n), mul(z, h));
  return { h: hNext, r, z, n };
}

/**
 * LSTM cell, PyTorch gate layout [i | f | g | o]:
 * c' = f ⊙ c + i ⊙ g,  h' = o ⊙ tanh(c')
 */
export function lstmCell(
  x: Var,
  h: Var,
  c: Var,
  wih: Var,
  whh: Var,
  bih?: Var,
  bhh?: Var,
): { h: Var; c: Var; i: Var; f: Var; g: Var; o: Var } {
  const H = h.shape[h.value.rank - 1];
  const gates = add(linear(x, wih, bih), linear(h, whh, bhh));
  const ax = gates.value.rank - 1;
  const i = sigmoid(slice(gates, ax, 0, H));
  const f = sigmoid(slice(gates, ax, H, 2 * H));
  const g = tanh(slice(gates, ax, 2 * H, 3 * H));
  const o = sigmoid(slice(gates, ax, 3 * H, 4 * H));
  const cNext = add(mul(f, c), mul(i, g));
  const hNext = mul(o, tanh(cNext));
  return { h: hNext, c: cNext, i, f, g, o };
}
