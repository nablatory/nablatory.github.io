import type { Trace, TraceNode } from '../runtime/trace';

/** What a block is, for drawing: data entering, a layer with weights, glue, a join, the loss. */
export type ModuleRole = 'input' | 'learned' | 'op' | 'junction' | 'objective' | 'output';

export interface Module {
  name: string;
  /** Nodes owned by the module, in execution order (parameters included). */
  nodes: TraceNode[];
  ops: TraceNode[];
  params: TraceNode[];
  output: TraceNode;
  firstId: number;
  inputs: string[];
  /** Inputs that appear later in execution order (recurrence). */
  feedback: string[];
  layer: number;
  row: number;
  paramCount: number;
  flops: number;
  /** ‖∂L/∂θ‖ over the module's parameters. */
  paramGradNorm: number;
  /** ‖∂L/∂y‖ at the module output. */
  outGradNorm: number;
  /** The block adds its input back to what it computed from it (a residual connection inside). */
  residual: boolean;
  kind: 'input' | 'op' | 'output';
  /** Name shown on screen; an input that shares a block's name is marked as data. */
  label: string;
  role: ModuleRole;
  /** Main operation, used for the glossary entry and the hover explanation. */
  category: string;
}

/** A tensor passed from one block to another. */
export interface ModuleEdge {
  from: string;
  to: string;
  /** The node whose value crosses the edge (the source block's output, or another of its values). */
  tensor: TraceNode;
  feedback: boolean;
  /** The value re-enters as a constant (stop-gradient): no gradient flows back along it. */
  stopGrad?: boolean;
}

export interface ModuleGraph {
  modules: Module[];
  byName: Map<string, Module>;
  moduleOf: Map<number, string>;
  edges: ModuleEdge[];
  layers: number;
}

const LEAF_OPS = new Set(['param', 'const', 'input']);

function l2(d?: ArrayLike<number>): number {
  if (!d) return 0;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

/** Operations that only rearrange values; an unscoped one belongs with the block that uses it. */
const REARRANGE = new Set(['reshape', 'transpose', 'slice', 'concat', 'pad', 'stack']);

/**
 * Group trace nodes into modules by the first segment of their scope.
 *
 * Inputs and named constants that nothing consumes (tensors attached to the output only for
 * display) are left out; unscoped rearrangements (reshape, concat, …) join the block that consumes them.
 */
export function buildModules(trace: Trace, depth = 1): ModuleGraph {
  const byId = new Map(trace.nodes.map((n) => [n.id, n]));
  const consumers = new Map<number, number[]>();
  for (const n of trace.nodes)
    for (const p of n.parents) {
      if (!consumers.has(p)) consumers.set(p, []);
      consumers.get(p)!.push(n.id);
    }
  const outputIds = new Set(Object.values(trace.outputs));
  const objectiveId = trace.objective?.node;
  const displayOnly = (n: TraceNode) =>
    !n.parents.length && n.op !== 'param' && !consumers.get(n.id)?.length && n.id !== objectiveId;

  const moduleOf = new Map<number, string>();
  const scopeName = (n: TraceNode) => n.scope.split('/').slice(0, depth).join('/');
  for (const n of trace.nodes) {
    if (displayOnly(n)) continue;
    if (n.op === 'input') moduleOf.set(n.id, n.key);
    else if (!LEAF_OPS.has(n.op)) moduleOf.set(n.id, n.scope ? scopeName(n) : n.key);
  }
  // Unscoped rearrangements join their first consumer's block (last first, so chains collapse).
  for (let i = trace.nodes.length - 1; i >= 0; i--) {
    const n = trace.nodes[i];
    if (n.scope || !REARRANGE.has(n.op) || outputIds.has(n.id) || n.id === objectiveId) continue;
    const c = consumers.get(n.id)?.[0];
    const m = c !== undefined ? moduleOf.get(c) : undefined;
    if (m) moduleOf.set(n.id, m);
  }
  // A parameter named after a block ("encoder.fc1.weight") belongs to it; other leaves
  // (parameters, constants) belong to their first consumer's module.
  const named = new Map([...moduleOf.values()].map((m) => [m.toLowerCase(), m]));
  for (const n of trace.nodes) {
    const owner = named.get(n.param?.split('.')[0].toLowerCase() ?? '');
    if (n.op === 'param' && owner) moduleOf.set(n.id, owner);
  }
  for (const n of trace.nodes) {
    if (moduleOf.has(n.id) || displayOnly(n)) continue;
    const c = consumers.get(n.id)?.[0];
    const m = c !== undefined ? moduleOf.get(c) : undefined;
    moduleOf.set(n.id, m ?? n.key);
  }

  const groups = new Map<string, TraceNode[]>();
  for (const n of trace.nodes) {
    const m = moduleOf.get(n.id);
    if (m === undefined) continue;
    if (!groups.has(m)) groups.set(m, []);
    groups.get(m)!.push(n);
  }

  const modules: Module[] = [];
  const crossing = new Map<string, Map<string, TraceNode>>();
  for (const [name, nodes] of groups) {
    const ops = nodes.filter((n) => !LEAF_OPS.has(n.op) || n.op === 'input');
    const params = nodes.filter((n) => n.op === 'param');
    if (!ops.length) continue;
    // Output: the last value passed on to another block; for a final block, its last declared
    // output (a view's tensor), else its last op.
    const leaves = (n: TraceNode) =>
      (consumers.get(n.id) ?? []).some((c) => moduleOf.get(c) !== name);
    const output =
      ops.findLast(leaves) ?? ops.findLast((n) => outputIds.has(n.id)) ?? ops[ops.length - 1];
    const inputs = new Map<string, TraceNode>();
    for (const n of nodes)
      for (const p of n.parents) {
        const pn = byId.get(p);
        if (!pn || pn.op === 'param' || pn.op === 'const') continue;
        const pm = moduleOf.get(p);
        if (pm !== undefined && pm !== name && !inputs.has(pm)) inputs.set(pm, pn);
      }
    crossing.set(name, inputs);
    const source = ops.every((n) => n.op === 'input' || !n.parents.length);
    // x + F(x) inside the block: an addition reads a tensor from outside that the block also
    // transforms.
    const inside = (id: number) => moduleOf.get(id) === name;
    const residual = ops.some(
      (n) =>
        n.op === 'add' &&
        n.parents.some((p) => {
          const pn = byId.get(p);
          if (!pn || inside(p) || pn.op === 'param' || pn.op === 'const') return false;
          return (consumers.get(p) ?? []).some((c) => c !== n.id && inside(c));
        }),
    );
    modules.push({
      name,
      nodes,
      ops,
      params,
      output,
      firstId: ops[0].id,
      inputs: [...inputs.keys()],
      feedback: [],
      layer: 0,
      row: 0,
      paramCount: params.reduce((s, p) => s + p.value.length, 0),
      flops: nodes.reduce((s, n) => s + n.flops, 0),
      paramGradNorm: Math.sqrt(params.reduce((s, p) => s + l2(p.grad) ** 2, 0)),
      outGradNorm: l2(output.grad),
      residual,
      kind: source ? 'input' : 'op',
      label: moduleLabel(name),
      role: 'op',
      category: 'op',
    });
  }
  modules.sort((a, b) => a.firstId - b.firstId);
  const byName = new Map(modules.map((m) => [m.name, m]));
  // The block holding the objective reports the objective, not whatever its scope computed last
  // (a per-step breakdown, say).
  const objectiveNode = objectiveId !== undefined ? byId.get(objectiveId) : undefined;
  const objectiveModule = objectiveNode ? byName.get(moduleOf.get(objectiveNode.id)!) : undefined;
  if (objectiveNode && objectiveModule) {
    objectiveModule.output = objectiveNode;
    objectiveModule.outGradNorm = l2(objectiveNode.grad);
  }

  // A value computed in one block and used in another with its gradient stopped (a target, say)
  // enters as a fresh constant named sg(…). The first node of another block holding the same
  // numbers is where it came from: the link is drawn, and that block reports the value it passes.
  const stopped = new Set<string>();
  const same = (a: TraceNode, b: TraceNode) =>
    a.value.length === b.value.length && a.value.every((v, i) => v === b.value[i]);
  for (const m of modules)
    for (const n of m.ops) {
      if (n.parents.length || !n.op.startsWith('sg(')) continue;
      let src: TraceNode | undefined;
      for (const k of trace.nodes) {
        if (k.id >= n.id) break;
        const km = k.parents.length ? moduleOf.get(k.id) : undefined;
        if (km !== undefined && km !== m.name && byName.has(km) && same(k, n)) {
          src = k;
          break;
        }
      }
      if (!src) continue;
      const from = moduleOf.get(src.id)!;
      const fm = byName.get(from)!;
      if (!crossing.get(m.name)!.has(from)) {
        crossing.get(m.name)!.set(from, n);
        m.inputs.push(from);
        stopped.add(`${from}→${m.name}`);
      }
      const passes = (x: TraceNode) =>
        outputIds.has(x.id) || (consumers.get(x.id) ?? []).some((c) => moduleOf.get(c) !== from);
      if (!passes(fm.output)) {
        fm.output = src;
        fm.outGradNorm = l2(src.grad);
      }
    }

  // Layer = longest path over forward edges; edges to earlier modules are feedback.
  const edges: ModuleEdge[] = [];
  for (const m of modules) {
    let layer = 0;
    const fwd: string[] = [];
    for (const i of m.inputs) {
      const pm = byName.get(i);
      if (!pm) continue;
      const tensor = crossing.get(m.name)!.get(i)!;
      if (pm.firstId < m.firstId) {
        layer = Math.max(layer, pm.layer + 1);
        fwd.push(i);
        edges.push({
          from: i,
          to: m.name,
          tensor,
          feedback: false,
          stopGrad: stopped.has(`${i}→${m.name}`),
        });
      } else {
        m.feedback.push(i);
        edges.push({ from: i, to: m.name, tensor, feedback: true });
      }
    }
    m.inputs = fwd;
    m.layer = layer;
  }
  const rows = new Map<number, number>();
  for (const m of modules) {
    m.row = rows.get(m.layer) ?? 0;
    rows.set(m.layer, m.row + 1);
  }
  const consumed = new Set(modules.flatMap((m) => m.inputs));
  const objective = objectiveId !== undefined ? moduleOf.get(objectiveId) : undefined;
  for (const m of modules) {
    if (m.kind !== 'input' && !consumed.has(m.name)) m.kind = 'output';
    m.category = categoryOf(m);
    m.role =
      m.kind === 'input'
        ? 'input'
        : m.name === objective || (m.kind === 'output' && m.category === 'loss')
          ? 'objective'
          : m.kind === 'output'
            ? 'output'
            : m.paramCount > 0
              ? 'learned'
              : JOIN.has(m.category) && m.inputs.length + m.feedback.length >= 2
                ? 'junction'
                : 'op';
  }
  // A reward given as data next to the block that predicts it: the input is marked as data.
  const blockLabels = new Set(modules.filter((m) => m.kind !== 'input').map((m) => m.label));
  for (const m of modules) if (m.kind === 'input' && blockLabels.has(m.label)) m.label += ' (data)';
  return {
    modules,
    byName,
    moduleOf,
    edges,
    layers: Math.max(0, ...modules.map((m) => m.layer)) + 1,
  };
}

const LOSS_OPS = new Set(['crossEntropy', 'mse', 'bceWithLogits', 'bernoulliNll']);
const ACTIVATIONS = new Set([
  'relu',
  'leakyRelu',
  'tanh',
  'sigmoid',
  'gelu',
  'geluTanh',
  'silu',
  'softplus',
  'elu',
  'scaledTanh',
]);
const JOIN = new Set(['add', 'concat', 'mul']);
const NOISE = /^(ε|epsilon|noise|z|z₀|mask)$/;

/** Names that say what a block is, checked before its operations. */
const NAMED: [RegExp, string][] = [
  [/^kl/i, 'kl'],
  [/target/i, 'targets'],
  [/return/i, 'returns'],
  [/^q$/i, 'value'],
  [/discreti/i, 'discretize'],
  [/^kernel$/i, 'kernel'],
  [/^(gcn|sage)\d*$/i, 'graphconv'],
  [/^gat\d*$/i, 'attention'],
  [/^conv$/i, 'conv'],
  [/sample|diffuse|corrupt/i, 'sample'],
  [/normali[sz]e/i, 'norm'],
  [/embed/i, 'embedding'],
];

/** The block's main operation: what a reader would call it. */
function categoryOf(m: Module): string {
  if (m.kind === 'input') return m.ops.some((n) => n.op === 'input') ? 'input' : 'source';
  const ops = new Set(m.ops.map((n) => n.op));
  const has = (...xs: string[]) => xs.some((x) => ops.has(x));
  const loud = [...ops].filter((o) => !QUIET.has(o) && !ACTIVATIONS.has(o));
  const leaves = m.ops.filter((n) => !n.parents.length);
  if (m.feedback.length) return 'recurrent';
  if ([...ops].some((o) => LOSS_OPS.has(o)) || (/loss/i.test(m.name) && m.kind === 'output'))
    return 'loss';
  for (const [re, c] of NAMED) if (re.test(m.name)) return c;
  if (/^(gates|state|cell)$/.test(m.name)) return 'recurrent';
  if (leaves.length && leaves.every((n) => n.op.startsWith('sg('))) return 'stopgrad';
  if (has('selectiveScan')) return 'ssm';
  if (has('softmax') && has('matmul')) return 'attention';
  if (has('conv2d')) return 'conv';
  if (has('convTranspose2d')) return 'deconv';
  if (has('axpby') || /^ode/.test(m.name)) return 'ode';
  if (has('linear') || (has('matmul') && m.paramCount > 0)) return 'affine';
  if (has('gather')) return 'embedding';
  if (has('layerNorm', 'batchNorm', 'rmsNorm')) return 'norm';
  if (has('maxPool2d', 'avgPool2d') || /pool/.test(m.name)) return 'pool';
  if (has('upsampleNearest2d')) return 'upsample';
  if (has('subsample2d')) return 'subsample';
  if (leaves.some((n) => NOISE.test(n.op))) return 'sample';
  if (has('softmax', 'logSoftmax')) return 'softmax';
  if (loud.length && loud.every((o) => o === 'add' || o === 'sub' || o === 'mul'))
    return has('add', 'sub') ? 'add' : 'mul';
  if (!loud.length && ops.has('concat')) return 'concat';
  if (!loud.length && [...ops].some((o) => ACTIVATIONS.has(o))) return 'activation';
  if (!loud.length) return 'reshape';
  return loud.includes('mean') ? 'mean' : 'op';
}

const OP_LABELS: Record<string, string> = {
  conv2d: 'conv',
  convTranspose2d: 'conv⊤',
  linear: 'affine',
  matmul: 'matmul',
  maxPool2d: 'max pool',
  avgPool2d: 'avg pool',
  upsampleNearest2d: 'upsample',
  layerNorm: 'layer norm',
  rmsNorm: 'RMS norm',
  batchNorm: 'batch norm',
  softmax: 'softmax',
  logSoftmax: 'log-softmax',
  crossEntropy: 'cross-entropy',
  mse: 'MSE',
  bceWithLogits: 'BCE',
  maskedFill: 'mask',
  gather: 'embedding',
  dropout: 'dropout',
};

const QUIET = new Set([
  'reshape',
  'transpose',
  'slice',
  'concat',
  'param',
  'const',
  'input',
  'scale',
  'addScalar',
  'pad',
  'neg',
]);

/** Short description of what a module computes, from its distinct op types. */
export function describeModule(m: Module): string {
  if (m.kind === 'input') return m.category === 'source' ? 'given tensor' : 'input';
  const seen: string[] = [];
  for (const n of m.ops) {
    if (QUIET.has(n.op)) continue;
    const label = OP_LABELS[n.op] ?? n.op;
    if (!seen.includes(label)) seen.push(label);
  }
  if (!seen.length) {
    const distinct = m.ops.map((n) => n.op).filter((v, i, a) => a.indexOf(v) === i);
    const main = distinct.filter((o) => !['reshape', 'transpose', 'slice'].includes(o));
    return (main.length ? main : distinct).slice(0, 2).join(' · ');
  }
  return seen.length > 3 ? `${seen.slice(0, 3).join(' · ')} …` : seen.join(' · ');
}

export function moduleLabel(name: string): string {
  if (name.startsWith('input:')) return name.slice(6);
  return name;
}
