import { Tensor, fmtShape, shapeEq, getDType } from './tensor';
import type { Rng } from './rng';
import type { ParamStore } from './params';

/** Vector–Jacobian product: maps ∂L/∂out to ∂L/∂input for each parent (null = no gradient). */
export type Backward = (grad: Tensor) => (Tensor | null)[];

export type AttrValue = number | string | boolean | number[] | null;
export type Attrs = Record<string, AttrValue>;

/** Replace selected elements of a node's output (activation patching). */
export interface ValueEdit {
  index: number;
  value: number;
}

export class Var {
  grad: Tensor | null = null;
  private gradOwned = false;

  constructor(
    readonly id: number,
    readonly op: string,
    readonly key: string,
    readonly scope: string,
    public value: Tensor,
    readonly parents: Var[],
    readonly parentIds: number[],
    readonly backwardFn: Backward | null,
    readonly requiresGrad: boolean,
    readonly attrs: Attrs,
    readonly aux: Record<string, Tensor> | undefined,
    readonly param: string | undefined,
    readonly edited: boolean,
  ) {}

  get shape(): number[] {
    return this.value.shape;
  }

  accumulate(g: Tensor): void {
    if (!shapeEq(g.shape, this.value.shape)) {
      throw new Error(
        `gradient shape ${fmtShape(g.shape)} != value shape ${fmtShape(this.value.shape)} at ${this.key}`,
      );
    }
    if (this.grad === null) {
      this.grad = g;
      this.gradOwned = false;
      return;
    }
    if (!this.gradOwned) {
      this.grad = this.grad.clone();
      this.gradOwned = true;
    }
    const d = this.grad.data;
    const s = g.data;
    for (let i = 0; i < d.length; i++) d[i] += s[i];
  }
}

export interface GraphOptions {
  /** Keep every node for inspection. */
  record?: boolean;
  /** Build backward closures. */
  grad?: boolean;
  /** Training-mode behaviour for dropout and batch norm. */
  training?: boolean;
  params?: ParamStore;
  /** Parameter names excluded from gradient computation. */
  frozen?: ReadonlySet<string>;
  /** Edits applied to the output of the node with the given key. */
  overrides?: ReadonlyMap<string, ValueEdit[]>;
  rng?: Rng;
}

let active: Graph | null = null;

export function activeGraph(): Graph | null {
  return active;
}

/**
 * A define-by-run computation record. Operations executed while a graph is
 * active are appended to it; `backward` walks them in reverse order.
 */
export class Graph {
  readonly nodes: Var[] = [];
  readonly record: boolean;
  readonly grad: boolean;
  readonly training: boolean;
  readonly params: ParamStore | undefined;
  readonly frozen: ReadonlySet<string>;
  readonly overrides: ReadonlyMap<string, ValueEdit[]>;
  readonly rng: Rng | undefined;
  private nextId = 0;
  private scopes: string[] = [];
  private keyCounts = new Map<string, number>();
  private paramVars = new Map<string, Var>();
  private gradPaused = 0;

  constructor(opts: GraphOptions = {}) {
    this.record = opts.record ?? true;
    this.grad = opts.grad ?? true;
    this.training = opts.training ?? false;
    this.params = opts.params;
    this.frozen = opts.frozen ?? new Set();
    this.overrides = opts.overrides ?? new Map();
    this.rng = opts.rng;
  }

  run<T>(fn: () => T): T {
    const prev = active;
    // The active graph is module state so ops can record onto it without threading it through.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    active = this;
    try {
      return fn();
    } finally {
      active = prev;
    }
  }

  get gradEnabled(): boolean {
    return this.grad && this.gradPaused === 0;
  }

  get scope(): string {
    return this.scopes.join('/');
  }

  withScope<T>(name: string, fn: () => T): T {
    this.scopes.push(name);
    try {
      return fn();
    } finally {
      this.scopes.pop();
    }
  }

  pauseGrad<T>(fn: () => T): T {
    this.gradPaused++;
    try {
      return fn();
    } finally {
      this.gradPaused--;
    }
  }

  private makeKey(op: string): string {
    const base = this.scope ? `${this.scope}/${op}` : op;
    const n = this.keyCounts.get(base) ?? 0;
    this.keyCounts.set(base, n + 1);
    return n === 0 ? base : `${base}#${n}`;
  }

  add(
    op: string,
    value: Tensor,
    parents: Var[],
    backward: Backward | null,
    attrs: Attrs = {},
    aux?: Record<string, Tensor>,
    opts: { key?: string; requiresGrad?: boolean; param?: string } = {},
  ): Var {
    const key = opts.key ?? this.makeKey(op);
    const requiresGrad =
      opts.requiresGrad ?? (this.gradEnabled && parents.some((p) => p.requiresGrad));

    let out = value;
    let bw = backward;
    const edits = this.overrides.get(key);
    if (edits && edits.length) {
      out = value.clone();
      for (const e of edits) {
        if (e.index >= 0 && e.index < out.size) out.data[e.index] = e.value;
      }
      if (bw) {
        const inner = bw;
        bw = (g) => {
          const masked = g.clone();
          for (const e of edits)
            if (e.index >= 0 && e.index < masked.size) masked.data[e.index] = 0;
          return inner(masked);
        };
      }
    }

    const keepParents = requiresGrad || this.record;
    const v = new Var(
      this.nextId++,
      op,
      key,
      this.scope,
      out,
      keepParents ? parents : [],
      parents.map((p) => p.id),
      requiresGrad ? bw : null,
      requiresGrad,
      attrs,
      aux,
      opts.param,
      !!edits?.length,
    );
    if (this.record) this.nodes.push(v);
    return v;
  }

  /** Parameter leaf. Repeated calls in one graph return the same node. */
  param(name: string): Var {
    const cached = this.paramVars.get(name);
    if (cached) return cached;
    if (!this.params) throw new Error(`graph has no parameter store (requested ${name})`);
    const t = this.params.get(name);
    const trainable = this.params.isTrainable(name) && !this.frozen.has(name);
    const v = this.add('param', t, [], null, {}, undefined, {
      key: `param:${name}`,
      requiresGrad: this.grad && trainable,
      param: name,
    });
    this.paramVars.set(name, v);
    return v;
  }

  paramVar(name: string): Var | undefined {
    return this.paramVars.get(name);
  }

  input(name: string, t: Tensor, requiresGrad = false): Var {
    return this.add('input', matchDType(t), [], null, {}, undefined, {
      key: `input:${name}`,
      requiresGrad: this.grad && requiresGrad,
    });
  }

  constant(t: Tensor, name = 'const'): Var {
    return this.add(name, matchDType(t), [], null, {}, undefined, { requiresGrad: false });
  }

  find(key: string): Var | undefined {
    return this.nodes.find((n) => n.key === key);
  }
}

function matchDType(t: Tensor): Tensor {
  const want64 = getDType() === 'float64';
  return want64 === t.data instanceof Float64Array ? t : t.cast();
}

/** Leaf that is not attached to any graph's parameters (used outside forward passes). */
export function leaf(t: Tensor, requiresGrad = false, name = 'leaf'): Var {
  const g = active;
  if (g)
    return g.add(name, matchDType(t), [], null, {}, undefined, {
      requiresGrad: g.grad && requiresGrad,
    });
  return new Var(
    -1,
    name,
    name,
    '',
    t,
    [],
    [],
    null,
    requiresGrad,
    {},
    undefined,
    undefined,
    false,
  );
}

/** Record an operation on the active graph (or produce an unrecorded value). */
export function op(
  name: string,
  value: Tensor,
  parents: Var[],
  backward: Backward,
  attrs: Attrs = {},
  aux?: Record<string, Tensor>,
): Var {
  const g = active;
  if (g) return g.add(name, value, parents, backward, attrs, aux);
  return new Var(-1, name, name, '', value, [], [], null, false, attrs, aux, undefined, false);
}

export function scope<T>(name: string, fn: () => T): T {
  const g = active;
  return g ? g.withScope(name, fn) : fn();
}

export function noGrad<T>(fn: () => T): T {
  const g = active;
  return g ? g.pauseGrad(fn) : fn();
}

export function isTraining(): boolean {
  return active?.training ?? false;
}

export function graphRng(): Rng {
  const r = active?.rng;
  if (!r) throw new Error('operation needs a random generator but the graph has none');
  return r;
}

/**
 * Reverse-mode differentiation. Every node that depends on a trainable leaf
 * receives ∂loss/∂node in `.grad`, including intermediate nodes, so upstream
 * gradients of any operation can be inspected afterwards.
 */
export function backward(loss: Var, seed?: Tensor): void {
  if (!loss.requiresGrad) throw new Error('loss does not depend on any trainable value');
  const order: Var[] = [];
  const seen = new Set<Var>();
  const stack: Var[] = [loss];
  while (stack.length) {
    const v = stack.pop()!;
    if (seen.has(v)) continue;
    seen.add(v);
    order.push(v);
    for (const p of v.parents) if (p.requiresGrad && !seen.has(p)) stack.push(p);
  }
  order.sort((a, b) => b.id - a.id);
  loss.accumulate(seed ?? Tensor.full(loss.shape, 1, loss.value.data));
  for (const v of order) {
    if (!v.grad || !v.backwardFn) continue;
    const gs = v.backwardFn(v.grad);
    for (let i = 0; i < v.parents.length; i++) {
      const p = v.parents[i];
      const g = gs[i];
      if (g && p.requiresGrad) p.accumulate(g);
    }
  }
}
