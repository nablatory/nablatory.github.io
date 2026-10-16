import { describe, expect, it } from 'vitest';
import { layoutGraph } from '../src/viz/graphLayout';
import { buildModules } from '../src/viz/architecture';
import { networkLayout, FULL, COMPACT } from '../src/viz/networkLayout';
import { MODELS } from '../src/models/catalog';
import { Session } from '../src/runtime/session';
import { fsLoader } from './helpers';
import { GLOSSARY } from '../src/ui/glossary';

const box = (id: string, w = 100, h = 40) => ({ id, w, h });

describe('layoutGraph', () => {
  it('stacks a chain top to bottom in one column', () => {
    const l = layoutGraph(
      [box('a'), box('b'), box('c')],
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
      ],
    );
    const [a, b, c] = ['a', 'b', 'c'].map((id) => l.nodes.get(id)!);
    expect(a.y).toBeLessThan(b.y);
    expect(b.y).toBeLessThan(c.y);
    expect(a.x).toBeCloseTo(b.x, 6);
    expect(b.x).toBeCloseTo(c.x, 6);
  });

  it('routes a skip edge through one point per crossed layer', () => {
    const l = layoutGraph(
      [box('a'), box('b'), box('c'), box('d')],
      [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'c' },
        { from: 'c', to: 'd' },
        { from: 'a', to: 'd' },
      ],
    );
    const skip = l.edges.find((e) => e.from === 'a' && e.to === 'd')!;
    // Port, two dummies (top and bottom of each), port.
    expect(skip.points).toHaveLength(2 + 2 * 2);
    // The skip lane does not run through the boxes it passes.
    for (const id of ['b', 'c']) {
      const n = l.nodes.get(id)!;
      const p = skip.points.find((q) => q.y >= n.y && q.y <= n.y + n.h)!;
      expect(p.x < n.x || p.x > n.x + n.w).toBe(true);
    }
  });

  it('places a source just above its consumer', () => {
    const l = layoutGraph(
      [box('x'), box('h1'), box('h2'), box('labels'), box('loss')],
      [
        { from: 'x', to: 'h1' },
        { from: 'h1', to: 'h2' },
        { from: 'h2', to: 'loss' },
        { from: 'labels', to: 'loss' },
      ],
    );
    expect(l.nodes.get('labels')!.layer).toBe(l.nodes.get('loss')!.layer - 1);
  });

  it('keeps feedback edges out of the layering and routes them on the right', () => {
    const l = layoutGraph(
      [box('x'), box('gates'), box('state'), box('out')],
      [
        { from: 'x', to: 'gates' },
        { from: 'gates', to: 'state' },
        { from: 'state', to: 'out' },
        { from: 'state', to: 'gates', feedback: true },
      ],
      { laneLabel: 80 },
    );
    const fb = l.edges.find((e) => e.feedback)!;
    const right = Math.max(...[...l.nodes.values()].map((n) => n.x + n.w));
    expect(fb.points[1].x).toBeGreaterThan(right);
    expect(l.width).toBeGreaterThan(fb.points[1].x + 80);
  });

  it('untangles a crossing', () => {
    // Initial order puts a's child to the right of b's child.
    const l = layoutGraph(
      [box('a'), box('b'), box('bb'), box('aa')],
      [
        { from: 'a', to: 'aa' },
        { from: 'b', to: 'bb' },
      ],
    );
    const a = l.nodes.get('a')!;
    const b = l.nodes.get('b')!;
    const aa = l.nodes.get('aa')!;
    const bb = l.nodes.get('bb')!;
    expect(Math.sign(a.x - b.x)).toBe(Math.sign(aa.x - bb.x));
  });
});

describe('module graph', () => {
  it('marks blocks that add their input back (residual connections)', async () => {
    const model = MODELS.find((m) => m.id === 'resmlp')!;
    const s = new Session(model);
    await s.loadData(fsLoader);
    const blocks = (g: ReturnType<typeof buildModules>) =>
      g.modules.filter((m) => /^block\d+$/.test(m.name));
    const on = blocks(buildModules(s.inspect({ kind: 'loss' })));
    expect(on.length).toBeGreaterThan(0);
    expect(on.every((m) => m.residual)).toBe(true);
    s.setConfig({ residual: false });
    const off = blocks(buildModules(s.inspect({ kind: 'loss' })));
    expect(off.length).toBe(on.length);
    expect(off.some((m) => m.residual)).toBe(false);
  });
});

describe('network layout of every model', () => {
  for (const model of MODELS) {
    it(`${model.id}: every block placed, no overlaps, edges join their blocks`, async () => {
      const s = new Session(model);
      await s.loadData(fsLoader);
      const trace = s.inspect({ kind: 'loss' });
      const g = buildModules(trace);
      // The block holding the objective shows the objective's value.
      if (trace.objective) {
        const holder = g.byName.get(g.moduleOf.get(trace.objective.node)!)!;
        expect(holder.output.id).toBe(trace.objective.node);
        expect(holder.role).toBe('objective');
      }
      // No stray rearrangement blocks (reshape#12 and the like) and no display-only constants.
      expect(g.modules.every((m) => !/#\d+$/.test(m.name) || m.role !== 'op')).toBe(true);
      // Every block has a hover explanation.
      for (const m of g.modules) expect(GLOSSARY[m.category], m.category).toBeDefined();
      for (const metrics of [FULL, COMPACT]) {
        const l = networkLayout(g, metrics);
        expect([...l.nodes.keys()].sort()).toEqual(
          [...g.modules.map((m) => m.name), ...l.ghosts.keys()].sort(),
        );
        // A ghost repeats an input next to one distant use.
        for (const src of l.ghosts.values()) expect(g.byName.get(src)!.role).toBe('input');
        const placed = [...l.nodes.values()];
        for (const p of placed) {
          expect(Number.isFinite(p.x) && Number.isFinite(p.y)).toBe(true);
          expect(p.x).toBeGreaterThanOrEqual(0);
          expect(p.x + p.w).toBeLessThanOrEqual(l.width);
          expect(p.y + p.h).toBeLessThanOrEqual(l.height);
        }
        for (let i = 0; i < placed.length; i++)
          for (let j = i + 1; j < placed.length; j++) {
            const a = placed[i];
            const b = placed[j];
            const apart =
              a.x + a.w <= b.x + 0.5 ||
              b.x + b.w <= a.x + 0.5 ||
              a.y + a.h <= b.y + 0.5 ||
              b.y + b.h <= a.y + 0.5;
            expect(apart, `${a.id} overlaps ${b.id}`).toBe(true);
          }
        for (const e of l.edges) {
          const a = l.nodes.get(e.from)!;
          expect(a, e.from).toBeDefined();
          const b = l.nodes.get(e.to)!;
          const first = e.points[0];
          const last = e.points[e.points.length - 1];
          if (e.feedback) {
            expect(first.x).toBeCloseTo(a.x + a.w, 6);
            expect(last.x).toBeCloseTo(b.x + b.w, 6);
          } else {
            expect(first.y).toBeCloseTo(a.y + a.h, 6);
            expect(last.y).toBeCloseTo(b.y, 6);
            expect(first.x).toBeGreaterThanOrEqual(a.x);
            expect(first.x).toBeLessThanOrEqual(a.x + a.w);
            expect(last.x).toBeGreaterThanOrEqual(b.x);
            expect(last.x).toBeLessThanOrEqual(b.x + b.w);
            expect(b.layer).toBeGreaterThan(a.layer);
          }
        }
        expect(l.edges.length).toBe(g.edges.length);
        // Each ghost stands in for exactly one edge of its input.
        for (const id of l.ghosts.keys())
          expect(l.edges.filter((e) => e.from === id)).toHaveLength(1);
      }
    });
  }
});
