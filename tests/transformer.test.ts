import { describe, expect, it } from 'vitest';
import { Graph, Rng, Tensor, backward, leaf, sum, withDType } from '../src/engine';
import transformer from '../src/models/transformer/transformer';
import bert from '../src/models/transformer/bert';
import gpt, { sampleDistribution } from '../src/models/transformer/gpt';
import vit from '../src/models/transformer/vit';
import { geluTanh } from '../src/models/transformer/ops';
import { causalMask, sinusoidalEncoding } from '../src/models/transformer/attention';
import {
  GRAMMAR_VOCAB,
  MASK,
  PAD,
  SPECIAL,
  WORD_IDS,
  mlmCorrupt,
  sentence,
} from '../src/models/transformer/grammar';
import { VOCAB_CHARS } from '../src/models/data/text';
import { Session } from '../src/runtime/session';
import { fetchCheckpoint } from '../src/runtime/checkpoint';
import type { Trace } from '../src/runtime/trace';
import type { Config, ModelDef } from '../src/models/types';
import { fsLoader } from './helpers';

const noop = { shouldStop: () => false, progress: () => {} };

async function session(model: ModelDef, config: Partial<Config> = {}, seed = 1): Promise<Session> {
  const s = new Session(model, { config, seed });
  await s.loadData(fsLoader);
  return s;
}

const output = (trace: Trace, name: string) =>
  trace.nodes.find((n) => n.id === trace.outputs[name])!;

/** Checks that every row of a [.., Tq, Tk] attention tensor is a distribution, and returns it. */
function expectRowsSumToOne(node: { shape: number[]; value: Float32Array }): void {
  const Tk = node.shape[node.shape.length - 1];
  for (let r = 0; r < node.value.length / Tk; r++) {
    let s = 0;
    for (let k = 0; k < Tk; k++) {
      const v = node.value[r * Tk + k];
      expect(v).toBeGreaterThanOrEqual(0);
      s += v;
    }
    expect(Math.abs(s - 1)).toBeLessThan(1e-5);
  }
}

describe('transformer family: gradients', () => {
  const tiny: [ModelDef, Partial<Config>][] = [
    [transformer, { dModel: 8, heads: 2, layers: 1, length: 4 }],
    [transformer, { dModel: 8, heads: 2, layers: 1, length: 4, norm: 'pre' }],
    [bert, { dModel: 8, heads: 2, layers: 1 }],
    [gpt, { dModel: 8, heads: 2, layers: 1, context: 4 }],
    [gpt, { dModel: 8, heads: 2, layers: 1, context: 4, tie: false }],
    [vit, { dModel: 8, heads: 2, layers: 1, patch: '14' }],
  ];
  it.each(tiny)(
    '%o: backpropagation matches finite differences (float64)',
    async (model, config) => {
      const s = await session(model, config);
      s.batchSize = 2;
      const res = s.gradCheck(undefined, 3);
      expect(res.entries.length).toBeGreaterThan(20);
      expect(res.maxRelError).toBeLessThan(1e-5);
    },
  );

  it('tanh-form GELU: VJP matches central differences', () => {
    withDType('float64', () => {
      const xs = [-4, -2.5, -1, -0.3, 0, 0.2, 0.7, 1.5, 3, 5];
      const g = new Graph({ grad: true, record: false });
      const x = g.run(() => {
        const v = leaf(new Tensor([xs.length], xs), true);
        backward(sum(geluTanh(v)));
        return v;
      });
      const f = (u: number) =>
        new Graph({ grad: false }).run(() => geluTanh(leaf(new Tensor([1], [u]))).value.data[0]);
      xs.forEach((u, i) => {
        const fd = (f(u + 1e-6) - f(u - 1e-6)) / 2e-6;
        expect(Math.abs(x.grad!.data[i] - fd)).toBeLessThan(1e-8);
      });
      // Close to x·Φ(x).
      expect(f(1)).toBeCloseTo(0.8413447, 3);
    });
  });
});

describe('attention', () => {
  it('rows sum to one; the causal mask gives exactly zero weight to future keys', async () => {
    const s = await session(gpt, { dModel: 16, heads: 2, layers: 2, context: 16 });
    const t = s.inspect({ kind: 'none' });
    for (const name of ['attn1', 'attnLast']) {
      const a = output(t, name);
      expect(a.shape).toEqual([1, 2, 16, 16]);
      expectRowsSumToOne(a);
      for (let h = 0; h < 2; h++)
        for (let q = 0; q < 16; q++)
          for (let k = q + 1; k < 16; k++) expect(a.value[(h * 16 + q) * 16 + k]).toBe(0);
    }
    const m = causalMask(3);
    expect(Array.from(m.data)).toEqual([0, 1, 1, 0, 0, 1, 0, 0, 0]);
  });

  it('GPT: ∂ logits_t / ∂ h⁰_{t′} = 0 for every t′ > t, and ≠ 0 without the mask', async () => {
    const T = 12;
    const V = VOCAB_CHARS.length;
    const s = await session(gpt, { dModel: 16, heads: 2, layers: 2, context: T });
    for (const t of [0, 5, T - 2]) {
      const tr = s.inspect({ kind: 'output', name: 'logits', index: t * V + 7 });
      const g = output(tr, 'h0').grad!;
      const D = 16;
      for (let u = t + 1; u < T; u++) for (let j = 0; j < D; j++) expect(g[u * D + j]).toBe(0);
      let past = 0;
      for (let u = 0; u <= t; u++) for (let j = 0; j < D; j++) past += Math.abs(g[u * D + j]);
      expect(past).toBeGreaterThan(0);
    }
    s.setConfig({ mask: 'none' });
    const g = output(s.inspect({ kind: 'output', name: 'logits', index: 0 * V + 7 }), 'h0').grad!;
    let future = 0;
    for (let i = 16; i < g.length; i++) future += Math.abs(g[i]);
    expect(future).toBeGreaterThan(0);
  });

  it('Transformer decoder: output t does not depend on decoder inputs after t; cross-attention rows sum to one', async () => {
    const s = await session(transformer, { dModel: 16, heads: 2, layers: 2, length: 6 });
    const tr0 = s.inspect({ kind: 'none' });
    for (const name of ['encAttn', 'selfAttn', 'crossAttn']) expectRowsSumToOne(output(tr0, name));
    const V = 11;
    for (const t of [0, 3]) {
      const tr = s.inspect({ kind: 'output', name: 'logits', index: t * V + 2 });
      const g = output(tr, 'decoderInput').grad!;
      for (let u = t + 1; u < 6; u++) for (let j = 0; j < 16; j++) expect(g[u * 16 + j]).toBe(0);
    }
  });

  it('sinusoidal encoding matches the closed form', () => {
    const pe = sinusoidalEncoding(10, 8);
    expect(pe.get(3, 0)).toBeCloseTo(Math.sin(3), 6);
    expect(pe.get(3, 1)).toBeCloseTo(Math.cos(3), 6);
    expect(pe.get(5, 4)).toBeCloseTo(Math.sin(5 / Math.pow(10000, 4 / 8)), 6);
    expect(pe.get(5, 5)).toBeCloseTo(Math.cos(5 / Math.pow(10000, 4 / 8)), 6);
  });
});

describe('BERT', () => {
  it('80/10/10 masking rule: selection rate and replacement frequencies', () => {
    const T = 1000;
    const rng = new Rng(9);
    const ids = Array.from({ length: 200 * T }, (_, i) =>
      i % 50 === 0 ? PAD : WORD_IDS[rng.int(WORD_IDS.length)],
    );
    const c = mlmCorrupt(ids, T, new Rng(10), 0.15);
    let maskable = 0;
    const counts = [0, 0, 0, 0];
    ids.forEach((t, i) => {
      if (t >= SPECIAL.length) maskable++;
      else expect(c.kind[i]).toBe(0);
      counts[c.kind[i]]++;
    });
    const sel = counts[1] + counts[2] + counts[3];
    expect(sel / maskable).toBeGreaterThan(0.145);
    expect(sel / maskable).toBeLessThan(0.155);
    expect(counts[1] / sel).toBeCloseTo(0.8, 1);
    expect(Math.abs(counts[2] / sel - 0.1)).toBeLessThan(0.01);
    expect(Math.abs(counts[3] / sel - 0.1)).toBeLessThan(0.01);
    c.kind.forEach((k, i) => {
      if (k === 1) expect(c.input[i]).toBe(MASK);
      if (k === 2) expect(WORD_IDS).toContain(c.input[i]);
      if (k === 3) expect(c.input[i]).toBe(ids[i]);
      expect(c.target[i]).toBe(k ? ids[i] : -1);
    });
  });

  it('every sequence gets at least one selected position; masks come from the graph generator', async () => {
    const short = Array.from({ length: 100 * 4 }, () => WORD_IDS[0]);
    const c = mlmCorrupt(short, 4, new Rng(1), 0.01);
    for (let s = 0; s < 100; s++)
      expect(c.kind.slice(s * 4, s * 4 + 4).some((k) => k > 0)).toBe(true);

    const s = await session(bert, { dModel: 16, heads: 2, layers: 1 });
    const batch = s.dataset.train(new Rng(3), 8, s.config);
    const lossWith = (seed: number) =>
      new Graph({ params: s.params, grad: false, record: false, rng: new Rng(seed) }).run(
        () =>
          bert.forward(batch, s.config, { training: true, rng: new Rng(seed) }).loss!.value.data[0],
      );
    expect(lossWith(5)).toBe(lossWith(5));
    expect(lossWith(5)).not.toBe(lossWith(6));
  });

  it('padded keys receive zero attention; probes mask the agreement word', async () => {
    const s = await session(bert);
    const t = s.inspect({ kind: 'none' });
    const tokens = t.meta!.tokens!;
    expect(tokens).toContain('[MASK]');
    const a = output(t, 'attnLast');
    expectRowsSumToOne(a);
    const T = tokens.length;
    tokens.forEach((tok, k) => {
      if (tok !== '[PAD]') return;
      for (let r = 0; r < a.value.length / T; r++) expect(a.value[r * T + k]).toBe(0);
    });
    expect(output(t, 'mlmProbs').shape).toEqual([1, GRAMMAR_VOCAB.length]);
    const sent = sentence(new Rng(1));
    expect(sent.ids.length).toBe(16);
  });
});

describe('learning', () => {
  it('Transformer: training lowers the validation loss on digit reversal', async () => {
    const s = await session(transformer, {}, 2);
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 60; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(0.75 * before);
    const g = s.runProcedure('greedy', { example: 1 }, noop);
    expect(g.text).toMatch(/greedy/);
    expect(g.tensors.cross.shape).toEqual([4, 6, 6]);
  });

  it('BERT: training lowers the masked-token loss', async () => {
    const s = await session(bert, {}, 2);
    const before = s.evaluate()!.loss;
    for (let i = 0; i < 60; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(0.75 * before);
  });

  it('GPT: training from scratch lowers the loss', async () => {
    const s = await session(gpt, { dModel: 32, heads: 2, layers: 1, context: 32 }, 2);
    s.reinitialize(2);
    const before = s.evaluate()!.loss;
    expect(before).toBeCloseTo(Math.log(VOCAB_CHARS.length), 1);
    for (let i = 0; i < 40; i++) s.trainStep();
    expect(s.evaluate()!.loss).toBeLessThan(before - 0.8);
  });

  it('ViT: patches reproduce the image, and training lowers the loss on MNIST', async () => {
    const s = await session(vit, {}, 2);
    const t = s.inspect({ kind: 'none' });
    const img = output(t, 'input').value;
    const patches = output(t, 'patches');
    expect(patches.shape).toEqual([16, 7, 7]);
    for (let k = 0; k < 16; k++)
      for (let i = 0; i < 7; i++)
        for (let j = 0; j < 7; j++) {
          const y = Math.floor(k / 4) * 7 + i;
          const x = (k % 4) * 7 + j;
          expect(patches.value[k * 49 + i * 7 + j]).toBe(img[y * 28 + x]);
        }
    expect(output(t, 'clsAttention').shape).toEqual([1, 1, 4, 4]);
    expectRowsSumToOne(output(t, 'attnLast'));
    const small = await session(vit, { dModel: 16, heads: 2, layers: 1 }, 2);
    const before = small.evaluate()!.loss;
    for (let i = 0; i < 100; i++) small.trainStep();
    expect(small.evaluate()!.loss).toBeLessThan(0.75 * before);
  });

  it('ViT: a drawing is re-tokenised when the patch size changes', async () => {
    const s = await session(vit);
    s.setCustomInput(new Tensor([1, 1, 28, 28], new Float32Array(784).fill(0.5)));
    expect(s.custom!.meta!.tokens).toHaveLength(17);
    s.setConfig({ patch: '4' });
    expect(s.custom!.meta!.tokens).toHaveLength(50);
    const t = s.inspect({ kind: 'none' });
    expect(output(t, 'clsAttention').shape).toEqual([1, 1, 7, 7]);
  });
});

describe('checkpoints and sampling', () => {
  it('the shipped checkpoint loads and reproduces its recorded validation loss', async () => {
    const { file, data } = await fetchCheckpoint(fsLoader, 'weights/gpt-shakespeare');
    expect(data.byteLength).toBeLessThan(2 * 1024 * 1024);
    const s = await session(gpt);
    s.setConfig(file.config);
    s.loadCheckpoint('shakespeare', file.params, data);
    const loss = s.evaluate()!.loss;
    expect(loss).toBeCloseTo(file.provenance.metrics.validationLoss, 4);
    expect(loss).toBeLessThan(2.3);
    const out = s.runProcedure('sample', { length: 40, temperature: 0.8, topK: 10 }, noop);
    expect(out.text!.length).toBe('First Citizen:\n'.length + 40);
    expect(out.tensors.dist.shape).toEqual([40, VOCAB_CHARS.length]);
    expect(s.runProcedure('sample', { length: 40 }, noop).text).toBe(
      s.runProcedure('sample', { length: 40 }, noop).text,
    );
  });

  it('the shipped ViT checkpoint loads and reproduces its recorded validation loss', async () => {
    const { file, data } = await fetchCheckpoint(fsLoader, 'weights/vit-mnist');
    expect(data.byteLength).toBeLessThan(2 * 1024 * 1024);
    const s = await session(vit);
    s.setConfig(file.config);
    s.loadCheckpoint('mnist', file.params, data);
    const r = s.evaluate()!;
    expect(r.loss).toBeCloseTo(file.provenance.metrics.validationLoss, 4);
    expect(r.metrics.accuracy).toBe(file.provenance.metrics.validationAccuracy);
    expect(r.metrics.accuracy).toBeGreaterThan(0.9);
  });

  it('temperature and top-k shape the sampling distribution', () => {
    const logits = [2, 1, 0.5, 0, -1, 3];
    const q = sampleDistribution(logits, 1, 3);
    expect(Array.from(q).filter((v) => v > 0).length).toBe(3);
    expect(q.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(q[4]).toBe(0);
    const cold = sampleDistribution(logits, 0.05, 0);
    expect(cold[5]).toBeGreaterThan(0.999);
    const full = sampleDistribution(logits, 1, 0);
    expect(full[5] / full[0]).toBeCloseTo(Math.exp(1), 5);
  });

  it('parameter count matches V·d + T·d + N(12d² + 13d) + 2d with tied embeddings', () => {
    const s = new Session(gpt);
    const {
      dModel: d,
      layers: N,
      context: T,
    } = s.config as { dModel: number; layers: number; context: number };
    expect(s.params.count()).toBe(
      VOCAB_CHARS.length * d + T * d + N * (12 * d * d + 13 * d) + 2 * d,
    );
  });
});
