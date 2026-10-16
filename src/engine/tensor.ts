export type DType = 'float32' | 'float64';
export type Data = Float32Array | Float64Array;

let defaultDType: DType = 'float32';

export function getDType(): DType {
  return defaultDType;
}

/** Run `fn` with tensors allocated in `dtype`. Float64 is used for gradient checks. */
export function withDType<T>(dtype: DType, fn: () => T): T {
  const prev = defaultDType;
  defaultDType = dtype;
  try {
    return fn();
  } finally {
    defaultDType = prev;
  }
}

export function alloc(n: number, like?: Data): Data {
  if (like) return like instanceof Float64Array ? new Float64Array(n) : new Float32Array(n);
  return defaultDType === 'float64' ? new Float64Array(n) : new Float32Array(n);
}

export function sizeOf(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

export function stridesOf(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

export function shapeEq(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function fmtShape(shape: readonly number[]): string {
  return `[${shape.join(', ')}]`;
}

/** Dense row-major tensor. Images are NCHW. */
export class Tensor {
  readonly shape: number[];
  readonly data: Data;

  constructor(shape: readonly number[], data?: Data | ArrayLike<number>) {
    this.shape = [...shape];
    const n = sizeOf(shape);
    if (data === undefined) {
      this.data = alloc(n);
    } else if (data instanceof Float32Array || data instanceof Float64Array) {
      this.data = data;
    } else {
      this.data = alloc(n);
      for (let i = 0; i < n; i++) this.data[i] = data[i];
    }
    if (this.data.length !== n) {
      throw new Error(
        `Tensor data length ${this.data.length} does not match shape ${fmtShape(shape)}`,
      );
    }
  }

  get size(): number {
    return this.data.length;
  }

  get rank(): number {
    return this.shape.length;
  }

  static zeros(shape: readonly number[], like?: Data): Tensor {
    return new Tensor(shape, alloc(sizeOf(shape), like));
  }

  static full(shape: readonly number[], value: number, like?: Data): Tensor {
    const t = Tensor.zeros(shape, like);
    t.data.fill(value);
    return t;
  }

  static scalar(value: number): Tensor {
    return new Tensor([], [value]);
  }

  static from(shape: readonly number[], values: ArrayLike<number>): Tensor {
    return new Tensor(shape, values);
  }

  clone(): Tensor {
    return new Tensor(this.shape, this.data.slice());
  }

  /** Copy into the current default dtype (or the given one). */
  cast(dtype: DType = getDType()): Tensor {
    const d = dtype === 'float64' ? new Float64Array(this.data) : new Float32Array(this.data);
    return new Tensor(this.shape, d);
  }

  reshape(shape: readonly number[]): Tensor {
    const resolved = resolveShape(shape, this.size);
    return new Tensor(resolved, this.data);
  }

  item(): number {
    if (this.size !== 1) throw new Error(`item() on tensor of shape ${fmtShape(this.shape)}`);
    return this.data[0];
  }

  get(...idx: number[]): number {
    return this.data[this.offset(idx)];
  }

  set(idx: number[], value: number): void {
    this.data[this.offset(idx)] = value;
  }

  offset(idx: readonly number[]): number {
    if (idx.length !== this.rank) throw new Error(`index rank ${idx.length} != ${this.rank}`);
    let o = 0;
    let stride = 1;
    for (let i = this.rank - 1; i >= 0; i--) {
      const v = idx[i];
      if (v < 0 || v >= this.shape[i]) throw new Error(`index ${v} out of range for axis ${i}`);
      o += v * stride;
      stride *= this.shape[i];
    }
    return o;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }
}

export function resolveShape(shape: readonly number[], size: number): number[] {
  const out = [...shape];
  const neg = out.indexOf(-1);
  if (neg >= 0) {
    const known = out.reduce((a, d, i) => (i === neg ? a : a * d), 1);
    if (known === 0 || size % known !== 0) throw new Error(`cannot infer -1 in ${fmtShape(shape)}`);
    out[neg] = size / known;
  }
  if (sizeOf(out) !== size) {
    throw new Error(`cannot reshape ${size} elements to ${fmtShape(shape)}`);
  }
  return out;
}

export interface TensorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  absMax: number;
  l2: number;
  nonFinite: number;
}

export function stats(data: ArrayLike<number>): TensorStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sq = 0;
  let nonFinite = 0;
  let n = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) {
      nonFinite++;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sq += v * v;
    n++;
  }
  const mean = n ? sum / n : 0;
  // Second pass about the mean: E[x²] − E[x]² cancels badly when |mean| ≫ std.
  let dev = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isFinite(v)) dev += (v - mean) * (v - mean);
  }
  const variance = n ? dev / n : 0;
  return {
    min: n ? min : 0,
    max: n ? max : 0,
    mean,
    std: Math.sqrt(variance),
    absMax: n ? Math.max(Math.abs(min), Math.abs(max)) : 0,
    l2: Math.sqrt(sq),
    nonFinite,
  };
}
