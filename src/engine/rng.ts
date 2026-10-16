/**
 * Seeded pseudo-random generator: xoshiro128** seeded through splitmix32.
 * Every stochastic step in the engine (initialisation, dropout, sampling,
 * data shuffling) draws from an explicit Rng so runs are reproducible.
 */
export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;
  private spare: number | null = null;
  readonly seed: number;

  constructor(seed: number) {
    this.seed = seed >>> 0;
    let x = this.seed;
    const next = () => {
      x = (x + 0x9e3779b9) | 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s0 = next();
    this.s1 = next();
    this.s2 = next();
    this.s3 = next();
  }

  /** Uniform 32-bit unsigned integer. */
  u32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5), 7), 9) >>> 0;
    const t = this.s1 << 9;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1). */
  uniform(): number {
    return this.u32() / 4294967296;
  }

  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.uniform();
  }

  int(n: number): number {
    return Math.floor(this.uniform() * n);
  }

  /** Standard normal via Box–Muller. */
  normal(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = this.uniform();
    const v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /** Categorical sample from unnormalised non-negative weights. */
  categorical(p: ArrayLike<number>): number {
    let total = 0;
    for (let i = 0; i < p.length; i++) total += p[i];
    let u = this.uniform() * total;
    for (let i = 0; i < p.length; i++) {
      u -= p[i];
      if (u < 0) return i;
    }
    return p.length - 1;
  }

  /** Independent stream derived from this seed and a label. */
  fork(label: string): Rng {
    let h = this.seed ^ 0x811c9dc5;
    for (let i = 0; i < label.length; i++) {
      h = Math.imul(h ^ label.charCodeAt(i), 0x01000193);
    }
    return new Rng(h >>> 0);
  }
}

function rotl(x: number, k: number): number {
  return (x << k) | (x >>> (32 - k));
}
