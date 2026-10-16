import { describe, expect, it } from 'vitest';
import { Graph, Rng, Tensor, leaf } from '../src/engine';
import gcn from '../src/models/graph/gcn';
import graphsage, { neighbourWeights } from '../src/models/graph/graphsage';
import gat from '../src/models/graph/gat';
import { normalizeAdjacency } from '../src/models/graph/common';
import {
  KARATE_EDGES,
  adjacencyTensor,
  karateGraph,
  loadGraph,
  sbmGraph,
} from '../src/models/data/graphs';
import type { Batch, ModelDef } from '../src/models/types';
import { Session } from '../src/runtime/session';

const models = [gcn, graphsage, gat];

function run(model: ModelDef, s: Session, batch: Batch, training = false, seed = 1) {
  const g = new Graph({
    params: s.params,
    record: false,
    grad: false,
    training,
    rng: new Rng(seed),
  });
  return g.run(() => model.forward(batch, s.config, { training, rng: new Rng(seed) }));
}

/** Relabel nodes: new node i is old node perm[i]. */
function permuteBatch(b: Batch, perm: number[]): Batch {
  const n = perm.length;
  const x = b.inputs.x;
  const F = x.shape[1];
  const xp = Tensor.zeros(x.shape);
  const ap = Tensor.zeros([n, n]);
  const mp = Tensor.zeros([n]);
  const lp = Tensor.zeros([n]);
  for (let i = 0; i < n; i++) {
    for (let f = 0; f < F; f++) xp.data[i * F + f] = x.data[perm[i] * F + f];
    for (let j = 0; j < n; j++) ap.data[i * n + j] = b.inputs.adj.data[perm[i] * n + perm[j]];
    mp.data[i] = b.inputs.mask.data[perm[i]];
    lp.data[i] = b.inputs.labelled.data[perm[i]];
  }
  return {
    inputs: { x: xp, adj: ap, mask: mp, labelled: lp },
    labels: perm.map((p) => b.labels![p]),
  };
}

describe('graph datasets', () => {
  it('karate club matches Zachary (1977): 34 nodes, 78 edges, published degrees, 17/17 split', () => {
    const g = karateGraph();
    expect(g.n).toBe(34);
    expect(KARATE_EDGES.length).toBe(78);
    expect(new Set(KARATE_EDGES.map(([a, b]) => `${a}-${b}`)).size).toBe(78);
    const deg = new Array(34).fill(0);
    for (const [a, b] of KARATE_EDGES) {
      deg[a]++;
      deg[b]++;
    }
    expect(deg).toEqual([
      16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4,
      6, 12, 17,
    ]);
    expect(g.labels.filter((y) => y === 0).length).toBe(17);
    expect(g.labels[0]).toBe(0);
    expect(g.labels[33]).toBe(1);
  });

  it('block model is deterministic and has no isolated nodes', () => {
    const a = sbmGraph(1);
    const b = sbmGraph(1);
    expect(a.edges).toEqual(b.edges);
    const deg = new Array(a.n).fill(0);
    for (const [i, j] of a.edges) {
      deg[i]++;
      deg[j]++;
    }
    expect(Math.min(...deg)).toBeGreaterThan(0);
    // Planted partition: most edges stay inside a block.
    const inside = a.edges.filter(([i, j]) => a.labels[i] === a.labels[j]).length;
    expect(inside / a.edges.length).toBeGreaterThan(0.7);
  });
});

describe('graph neural networks', () => {
  const tiny: Record<string, Record<string, number | boolean>> = {
    gcn: { hidden: 4, layers: 3 },
    graphsage: { hidden: 4, layers: 3, embed2d: true },
    gat: { hidden: 2, heads: 2, outHeads: 2, layers: 2 },
  };

  it.each(models)('$id: backpropagation matches finite differences (float64)', (m) => {
    for (const graph of ['karate', 'sbm']) {
      const s = new Session(m, { config: { ...tiny[m.id], graph } });
      const res = s.gradCheck(undefined, 3);
      expect(res.maxRelError).toBeLessThan(1e-5);
    }
  });

  it('GraphSAGE max-pooling aggregator: gradients match finite differences', () => {
    const s = new Session(graphsage, { config: { hidden: 4, aggregator: 'pool', graph: 'sbm' } });
    const res = s.gradCheck(undefined, 3);
    expect(res.maxRelError).toBeLessThan(1e-5);
  });

  it('GCN normalised adjacency: symmetric, and rows sum to 1 on a regular graph', () => {
    // 6-cycle: 2-regular, so Ã = A + I has every degree 3 and Â = Ã / 3.
    const n = 6;
    const edges: [number, number][] = Array.from({ length: n }, (_, i) => [i, (i + 1) % n]);
    const A = adjacencyTensor(n, edges);
    const g = new Graph({ record: false, grad: false });
    const [sym, row] = g.run(() => [
      normalizeAdjacency(leaf(A), true, 'symmetric').value,
      normalizeAdjacency(leaf(A), true, 'row').value,
    ]);
    for (let i = 0; i < n; i++) {
      let rs = 0;
      for (let j = 0; j < n; j++) {
        expect(sym.data[i * n + j]).toBeCloseTo(sym.data[j * n + i], 7);
        rs += sym.data[i * n + j];
        const expected = i === j || A.data[i * n + j] ? 1 / 3 : 0;
        expect(sym.data[i * n + j]).toBeCloseTo(expected, 7);
        expect(row.data[i * n + j]).toBeCloseTo(expected, 7);
      }
      expect(rs).toBeCloseTo(1, 6);
    }
    // Irregular graph (karate): Â_ij = Ã_ij / √(d̃_i d̃_j), still symmetric.
    const K = adjacencyTensor(34, KARATE_EDGES);
    const k = new Graph({ record: false, grad: false }).run(
      () => normalizeAdjacency(leaf(K), true, 'symmetric').value,
    );
    expect(k.data[0 * 34 + 33]).toBe(0);
    expect(k.data[0 * 34 + 1]).toBeCloseTo(1 / Math.sqrt(17 * 10), 6);
    expect(k.data[0]).toBeCloseTo(1 / 17, 6);
    for (let i = 0; i < 34; i++)
      for (let j = 0; j < 34; j++) expect(k.data[i * 34 + j]).toBeCloseTo(k.data[j * 34 + i], 7);
  });

  it.each(models)(
    '$id is permutation-equivariant: permuting the nodes permutes the output',
    (m) => {
      const s = new Session(m, { config: { graph: 'sbm', dropout: 0 } });
      const batch = s.dataset.example(0, s.config);
      const perm = new Rng(9).shuffle(Array.from({ length: 60 }, (_, i) => i));
      const a = run(m, s, batch).outputs.logits.value;
      const b = run(m, s, permuteBatch(batch, perm)).outputs.logits.value;
      const C = a.shape[1];
      for (let i = 0; i < 60; i++)
        for (let c = 0; c < C; c++)
          expect(b.data[i * C + c]).toBeCloseTo(a.data[perm[i] * C + c], 4);
    },
  );

  it('GAT attention rows sum to 1 over N(i) ∪ {i} and are exactly 0 elsewhere', () => {
    const s = new Session(gat, { config: { heads: 3 } });
    const batch = s.dataset.example(0, s.config);
    const att = run(gat, s, batch).outputs.attention.value;
    const n = 34;
    expect(att.shape).toEqual([3, n, n]);
    const A = batch.inputs.adj.data;
    for (let h = 0; h < 3; h++)
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) {
          const v = att.data[(h * n + i) * n + j];
          if (i !== j && A[i * n + j] === 0) expect(v).toBe(0);
          else expect(v).toBeGreaterThan(0);
          sum += v;
        }
        expect(sum).toBeCloseTo(1, 6);
      }
  });

  it('GraphSAGE neighbourhood: mean weights are D⁻¹A; sampled weights stay on neighbours and sum to 1', () => {
    const A = adjacencyTensor(34, KARATE_EDGES);
    const full = new Graph({ record: false, grad: false }).run(() => neighbourWeights(A, 3));
    const sampled = new Graph({ record: false, grad: false, training: true, rng: new Rng(4) }).run(
      () => neighbourWeights(A, 3),
    );
    for (let i = 0; i < 34; i++) {
      let deg = 0;
      for (let j = 0; j < 34; j++) deg += A.data[i * 34 + j];
      let rs = 0;
      for (let j = 0; j < 34; j++) {
        expect(full.data[i * 34 + j]).toBeCloseTo(A.data[i * 34 + j] / deg, 7);
        if (A.data[i * 34 + j] === 0) expect(sampled.data[i * 34 + j]).toBe(0);
        rs += sampled.data[i * 34 + j];
      }
      expect(rs).toBeCloseTo(1, 6);
    }
  });

  it('GCN learns the karate-club factions from two labelled members per faction', () => {
    const s = new Session(gcn, { seed: 1 });
    const before = s.evaluate()!;
    for (let i = 0; i < 200; i++) s.trainStep();
    const after = s.evaluate()!;
    expect(after.loss).toBeLessThan(before.loss);
    // 30 unlabelled members.
    expect(after.metrics.unlabelledAccuracy).toBeGreaterThanOrEqual(0.85);
    expect(after.metrics.labelledAccuracy).toBe(1);
  });

  it.each([graphsage, gat])('$id learns the block model from two labelled nodes per block', (m) => {
    const s = new Session(m, { seed: 2, config: { graph: 'sbm' } });
    for (let i = 0; i < 200; i++) s.trainStep();
    expect(s.evaluate()!.metrics.unlabelledAccuracy).toBeGreaterThan(0.75);
  });

  it('views reference tensors the forward pass produces', () => {
    for (const m of models) {
      const s = new Session(m);
      const t = s.inspect({ kind: 'none' });
      for (const v of m.views) {
        const refs = 'source' in v ? [v.source] : [];
        if (v.kind === 'graph' && v.colorBy && v.colorBy !== 'argmax') refs.push(v.colorBy);
        if (v.kind === 'scatter' && v.colorBy) refs.push(v.colorBy);
        for (const r of refs)
          if (r.startsWith('out:')) expect(t.outputs[r.slice(4)], `${m.id} ${r}`).toBeDefined();
      }
      expect(t.meta?.edges).toBe(loadGraph('karate').edges);
    }
  });

  it('a forward pass and a training step are fast at the default configuration', () => {
    for (const m of models) {
      const s = new Session(m);
      for (let i = 0; i < 3; i++) s.trainStep();
      const t0 = performance.now();
      for (let i = 0; i < 10; i++) s.inspect({ kind: 'none' });
      const t1 = performance.now();
      for (let i = 0; i < 10; i++) s.trainStep();
      const t2 = performance.now();
      expect((t1 - t0) / 10).toBeLessThan(100);
      expect((t2 - t1) / 10).toBeLessThan(250);
    }
  });
});
