import { Tensor, sizeOf, type DType } from './tensor';
import type { Rng } from './rng';

export interface ParamInfo {
  /** Human-readable initializer, e.g. "He normal, σ = √(2/fan_in)". */
  init: string;
  trainable: boolean;
  /** Origin of the current values. */
  source: 'init' | 'pretrained' | 'trained' | 'edited';
}

export class ParamStore {
  private values = new Map<string, Tensor>();
  private info = new Map<string, ParamInfo>();

  add(name: string, t: Tensor, init: string, trainable = true): Tensor {
    if (this.values.has(name)) throw new Error(`duplicate parameter ${name}`);
    this.values.set(name, t);
    this.info.set(name, { init, trainable, source: 'init' });
    return t;
  }

  has(name: string): boolean {
    return this.values.has(name);
  }

  get(name: string): Tensor {
    const t = this.values.get(name);
    if (!t) throw new Error(`unknown parameter ${name}`);
    return t;
  }

  set(name: string, t: Tensor): void {
    const prev = this.get(name);
    if (prev.shape.length !== t.shape.length || prev.shape.some((d, i) => d !== t.shape[i]))
      throw new Error(
        `shape mismatch replacing ${name}: [${prev.shape.join(', ')}] → [${t.shape.join(', ')}]`,
      );
    this.values.set(name, t);
  }

  meta(name: string): ParamInfo {
    return this.info.get(name)!;
  }

  setSource(name: string, source: ParamInfo['source']): void {
    this.info.get(name)!.source = source;
  }

  isTrainable(name: string): boolean {
    return this.info.get(name)?.trainable ?? false;
  }

  names(): string[] {
    return [...this.values.keys()];
  }

  entries(): [string, Tensor][] {
    return [...this.values.entries()];
  }

  count(filter?: (name: string) => boolean): number {
    let n = 0;
    for (const [k, v] of this.values) if (!filter || filter(k)) n += v.size;
    return n;
  }

  clone(dtype?: DType): ParamStore {
    const out = new ParamStore();
    for (const [k, v] of this.values) {
      out.values.set(k, dtype ? v.cast(dtype) : v.clone());
      out.info.set(k, { ...this.info.get(k)! });
    }
    return out;
  }

  /** Flat serialisation: manifest of names/shapes plus one Float32Array. */
  serialize(): {
    manifest: { name: string; shape: number[]; offset: number }[];
    data: Float32Array;
  } {
    const manifest: { name: string; shape: number[]; offset: number }[] = [];
    let total = 0;
    for (const [k, v] of this.values) {
      manifest.push({ name: k, shape: v.shape, offset: total });
      total += v.size;
    }
    const data = new Float32Array(total);
    for (const m of manifest) data.set(this.values.get(m.name)!.data, m.offset);
    return { manifest, data };
  }

  /** Load values by name; every manifest entry must match an existing parameter's shape. */
  load(
    manifest: { name: string; shape: number[]; offset: number }[],
    data: Float32Array,
    source: ParamInfo['source'] = 'pretrained',
  ): void {
    for (const m of manifest) {
      const cur = this.values.get(m.name);
      if (!cur) throw new Error(`checkpoint parameter ${m.name} not in model`);
      if (cur.shape.join('x') !== m.shape.join('x')) {
        throw new Error(
          `checkpoint shape [${m.shape}] for ${m.name} does not match model [${cur.shape}]`,
        );
      }
      const n = sizeOf(m.shape);
      const t = Tensor.zeros(m.shape, cur.data);
      t.data.set(data.subarray(m.offset, m.offset + n));
      this.values.set(m.name, t);
      this.info.get(m.name)!.source = source;
    }
  }
}

export type Init =
  | { kind: 'zeros' }
  | { kind: 'ones' }
  | { kind: 'const'; value: number }
  | { kind: 'normal'; std: number; mean?: number }
  | { kind: 'uniform'; bound: number }
  | { kind: 'heNormal'; fanIn: number }
  | { kind: 'heUniform'; fanIn: number }
  | { kind: 'xavierUniform'; fanIn: number; fanOut: number }
  | { kind: 'xavierNormal'; fanIn: number; fanOut: number }
  | { kind: 'lecunUniform'; fanIn: number };

export function describeInit(init: Init): string {
  switch (init.kind) {
    case 'zeros':
      return 'zeros';
    case 'ones':
      return 'ones';
    case 'const':
      return `constant ${init.value}`;
    case 'normal':
      return `N(${init.mean ?? 0}, ${init.std}²)`;
    case 'uniform':
      return `U(−${fmt(init.bound)}, ${fmt(init.bound)})`;
    case 'heNormal':
      return `He normal, σ = √(2/${init.fanIn}) = ${fmt(Math.sqrt(2 / init.fanIn))}`;
    case 'heUniform':
      return `He uniform, bound √(6/${init.fanIn}) = ${fmt(Math.sqrt(6 / init.fanIn))}`;
    case 'xavierUniform':
      return `Glorot uniform, bound √(6/(${init.fanIn}+${init.fanOut}))`;
    case 'xavierNormal':
      return `Glorot normal, σ = √(2/(${init.fanIn}+${init.fanOut}))`;
    case 'lecunUniform':
      return `PyTorch default, U(±1/√${init.fanIn})`;
  }
}

function fmt(x: number): string {
  return x.toPrecision(3);
}

export function initTensor(shape: number[], init: Init, rng: Rng): Tensor {
  const t = Tensor.zeros(shape);
  const d = t.data;
  switch (init.kind) {
    case 'zeros':
      break;
    case 'ones':
      d.fill(1);
      break;
    case 'const':
      d.fill(init.value);
      break;
    case 'normal':
      for (let i = 0; i < d.length; i++) d[i] = (init.mean ?? 0) + init.std * rng.normal();
      break;
    case 'uniform':
      for (let i = 0; i < d.length; i++) d[i] = rng.range(-init.bound, init.bound);
      break;
    case 'heNormal': {
      const s = Math.sqrt(2 / init.fanIn);
      for (let i = 0; i < d.length; i++) d[i] = s * rng.normal();
      break;
    }
    case 'heUniform': {
      const b = Math.sqrt(6 / init.fanIn);
      for (let i = 0; i < d.length; i++) d[i] = rng.range(-b, b);
      break;
    }
    case 'xavierUniform': {
      const b = Math.sqrt(6 / (init.fanIn + init.fanOut));
      for (let i = 0; i < d.length; i++) d[i] = rng.range(-b, b);
      break;
    }
    case 'xavierNormal': {
      const s = Math.sqrt(2 / (init.fanIn + init.fanOut));
      for (let i = 0; i < d.length; i++) d[i] = s * rng.normal();
      break;
    }
    case 'lecunUniform': {
      const b = 1 / Math.sqrt(init.fanIn);
      for (let i = 0; i < d.length; i++) d[i] = rng.range(-b, b);
      break;
    }
  }
  return t;
}

/** Convenience for model `init` functions. */
export class ParamBuilder {
  readonly store = new ParamStore();
  constructor(readonly rng: Rng) {}

  add(name: string, shape: number[], init: Init, trainable = true): void {
    this.store.add(
      name,
      initTensor(shape, init, this.rng.fork(name)),
      describeInit(init),
      trainable,
    );
  }

  /** Fixed (non-trainable) tensor such as a positional encoding or normalised adjacency. */
  fixed(name: string, t: Tensor, description: string): void {
    this.store.add(name, t, description, false);
  }
}
