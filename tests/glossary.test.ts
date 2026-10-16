import { describe, expect, it } from 'vitest';
import { GLOSSARY, OP_TERMS, PROSE, annotate, findTerm, opTerm } from '../src/ui/glossary';

describe('glossary', () => {
  it('points every operation and prose phrase at an existing entry', () => {
    for (const [op, id] of Object.entries(OP_TERMS)) expect(GLOSSARY[id], op).toBeDefined();
    for (const [id] of PROSE) expect(GLOSSARY[id], id).toBeDefined();
    expect(opTerm('conv2d')).toBe('conv');
    expect(opTerm('noSuchOp')).toBeUndefined();
  });

  it('gives every entry a short explanation', () => {
    for (const [id, e] of Object.entries(GLOSSARY)) {
      expect(e.term.length, id).toBeGreaterThan(0);
      expect(e.what.length, id).toBeLessThanOrEqual(200);
      expect((e.why ?? '').length, id).toBeLessThanOrEqual(200);
    }
  });

  it('picks the entry with the longest match for a label', () => {
    expect(findTerm('KL weight β')).toBe('klWeight');
    expect(findTerm('KL')).toBe('kl');
    expect(findTerm('Reward coefficient c₁')).toBe('lossWeight');
    expect(findTerm('reward')).toBe('rewardLoss');
    expect(findTerm('Learning rate')).toBe('lr');
    expect(findTerm('Colour of the sky')).toBeUndefined();
  });

  it('marks the first occurrence of a term in prose, never inside a quoted name', () => {
    const parts = annotate('A “softmax” label; softmax again, then softmax.');
    const marked = parts.filter((p) => p.term);
    expect(marked).toHaveLength(1);
    expect(marked[0]).toEqual({ text: 'softmax', term: 'softmax' });
    expect(parts.map((p) => p.text).join('')).toBe(
      'A “softmax” label; softmax again, then softmax.',
    );
  });

  it('prefers the longer phrase and respects the limit', () => {
    const parts = annotate('Transposed convolutions undo a convolution.');
    expect(parts.find((p) => p.term)?.term).toBe('deconv');
    const many = annotate('softmax, dropout, ReLU, gradients, learning rate, MPPI', 3);
    expect(many.filter((p) => p.term)).toHaveLength(3);
  });
});
