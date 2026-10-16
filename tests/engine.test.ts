import { describe, expect, it } from 'vitest';
import { ParamStore, Tensor, stats } from '../src/engine';

describe('stats', () => {
  it('keeps the standard deviation when the mean is much larger than the spread', () => {
    // E[x²] − E[x]² loses every digit here in float64; the two-pass sum does not.
    const x = Float64Array.from({ length: 1000 }, (_, i) => 1e8 + (i % 2 ? 1 : -1));
    expect(stats(x).std).toBeCloseTo(1, 12);
  });

  it('ignores non-finite entries', () => {
    const s = stats([1, 3, NaN, Infinity]);
    expect(s.mean).toBe(2);
    expect(s.std).toBe(1);
  });
});

describe('ParamStore.set', () => {
  it('rejects a tensor of the same size but a different shape', () => {
    const p = new ParamStore();
    p.add('w', Tensor.zeros([2, 3]), 'zeros');
    expect(() => p.set('w', Tensor.zeros([3, 2]))).toThrow(/shape mismatch replacing w/);
    p.set('w', Tensor.full([2, 3], 1));
    expect(p.get('w').data[0]).toBe(1);
  });
});
