import type { Module, ModuleGraph } from './architecture';
import { describeModule } from './architecture';
import { layoutGraph, type Layout, type LayoutNode } from './graphLayout';
import type { TraceNode } from '../runtime/trace';

export interface NetworkMetrics {
  /** Block width and height. */
  w: number;
  h: number;
  /** Height of a loss block (no tensor tile). */
  hObjective: number;
  /** Side of the tensor tile. */
  tile: number;
  /** Diameter of a join (⊕) node. */
  join: number;
  /** Height of one row in a block's list of operations. */
  opRow: number;
  /** Height of a reference to an input drawn again next to a distant use. */
  ghost: number;
}

export const FULL: NetworkMetrics = {
  w: 224,
  h: 60,
  hObjective: 48,
  tile: 44,
  join: 30,
  opRow: 18,
  ghost: 24,
};
export const COMPACT: NetworkMetrics = {
  w: 168,
  h: 56,
  hObjective: 46,
  tile: 36,
  join: 28,
  opRow: 18,
  ghost: 22,
};

/** The part of a 2-D canvas context used for measuring text (also typed without the DOM). */
interface Measurer {
  font: string;
  measureText(s: string): { width: number };
}
let ctx: Measurer | null | undefined;

/** Width of a label in IBM Plex Sans (measured in a browser, estimated elsewhere). */
export function textWidth(s: string, px = 11.5, weight = 400, mono = false): number {
  if (ctx === undefined) {
    const doc = (
      globalThis as {
        document?: { createElement(t: 'canvas'): { getContext(k: '2d'): Measurer | null } };
      }
    ).document;
    ctx = doc ? doc.createElement('canvas').getContext('2d') : null;
  }
  if (!ctx) return s.length * px * (mono ? 0.6 : 0.56);
  ctx.font = mono
    ? `${weight} ${px}px 'IBM Plex Mono', ui-monospace, monospace`
    : `${weight} ${px}px 'IBM Plex Sans Variable', 'IBM Plex Sans', system-ui, sans-serif`;
  return ctx.measureText(s).width;
}

export function visibleOps(m: Module): TraceNode[] {
  return [...m.params, ...m.ops.filter((n) => n.op !== 'const')];
}

export function joinLabelWidth(m: Module): number {
  return Math.min(
    132,
    Math.max(textWidth(m.label, 12.5, 600), textWidth(describeModule(m), 11)) + 10,
  );
}

/** Ghost id for an input drawn again above a distant block that uses it. */
export const ghostId = (from: string, to: string) => `↳${from}→${to}`;

export interface NetworkLayout extends Layout {
  /** Ghost node id → the input block it repeats. */
  ghosts: Map<string, string>;
}

/** An input reused this many layers further down is drawn again next to that use. */
const GHOST_SPAN = 3;

/** Box size of each block, then the layered layout of the graph. */
export function networkLayout(
  graph: ModuleGraph,
  metrics: NetworkMetrics = FULL,
  expanded: ReadonlySet<string> = new Set(),
): NetworkLayout {
  const nodes: LayoutNode[] = graph.modules.map((m) => {
    if (m.role === 'junction' && !expanded.has(m.name))
      return {
        id: m.name,
        w: metrics.join + 8 + joinLabelWidth(m),
        h: metrics.join,
        anchor: metrics.join / 2,
      };
    const base = m.role === 'objective' ? metrics.hObjective : metrics.h;
    const extra = expanded.has(m.name) ? 8 + visibleOps(m).length * metrics.opRow : 0;
    return { id: m.name, w: metrics.w, h: base + extra };
  });
  const opts = {
    layerGap: 40,
    nodeGap: metrics === COMPACT ? 22 : 28,
    dummyGap: 14,
    pad: 20,
    laneGap: 18,
    laneLabel: 0,
  };
  const edges = graph.edges.map((e) => ({ from: e.from, to: e.to, feedback: e.feedback }));
  const first = layoutGraph(nodes, edges, opts);
  // Inputs used again far below (an image compared in the loss, say) get a small reference
  // above that use instead of a line across the whole graph.
  const ghosts = new Map<string, string>();
  const fanout = new Map<string, number>();
  for (const e of edges) if (!e.feedback) fanout.set(e.from, (fanout.get(e.from) ?? 0) + 1);
  const rewired = edges.map((e) => {
    const src = graph.byName.get(e.from);
    const a = first.nodes.get(e.from);
    const b = first.nodes.get(e.to);
    if (
      e.feedback ||
      !src ||
      src.role !== 'input' ||
      (fanout.get(e.from) ?? 0) < 2 ||
      !a ||
      !b ||
      b.layer - a.layer < GHOST_SPAN
    )
      return e;
    const id = ghostId(e.from, e.to);
    ghosts.set(id, e.from);
    nodes.push({
      id,
      w: Math.ceil(textWidth(src.label, 11, 500)) + 30,
      h: metrics.ghost,
    });
    return { from: id, to: e.to, feedback: false };
  });
  if (!ghosts.size) return { ...first, ghosts };
  return { ...layoutGraph(nodes, rewired, opts), ghosts };
}
