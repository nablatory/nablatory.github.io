import { describe, expect, it } from 'vitest';
import { MODELS } from '../src/models/catalog';
import type { ConfigField, ConfigValue, ModelDef } from '../src/models/types';
import { Session, type TrainRecord } from '../src/runtime/session';
import { fsLoader } from './helpers';

/** Buttons of the Guide that its text may name. */
const BUTTONS = ['Start from random weights', 'Load trained weights'];

function allowed(f: ConfigField, v: ConfigValue): boolean {
  switch (f.kind) {
    case 'select':
      return f.options.some((o) => o.value === v);
    case 'bool':
      return typeof v === 'boolean';
    case 'text':
      return typeof v === 'string' && !f.validate?.(v);
    default:
      return (
        typeof v === 'number' &&
        v >= f.min &&
        v <= f.max &&
        (f.kind !== 'int' || Number.isInteger(v))
      );
  }
}

/** Every “quoted” name in the text is a button, procedure, view title (or its start) or setting. */
function unknownQuotes(m: ModelDef, text: string): string[] {
  const names = [
    ...BUTTONS,
    ...(m.procedures ?? []).map((p) => p.label),
    ...m.configSchema.map((f) => f.label),
  ];
  return [...text.matchAll(/“([^”]+)”/g)]
    .map((x) => x[1])
    .filter(
      (q) =>
        !names.includes(q) &&
        !m.views.some((v) => v.title.startsWith(q)) &&
        // Quoted symbols and sample text (“?”, “·”, example sentences) are not names.
        !/^[^A-Za-z]*$/.test(q) &&
        !q.includes(' . ') &&
        !q.startsWith('f depends') &&
        !q.startsWith('sampled') &&
        !q.startsWith('relative kernel') &&
        q !== 'none',
    );
}

describe('guide text', () => {
  for (const m of MODELS) {
    describe(m.id, () => {
      it('exists, names real views, settings and buttons, and suggests valid changes', () => {
        const g = m.guide;
        expect(g, 'guide').toBeDefined();
        expect(g!.what.split(/\s+/).length).toBeGreaterThan(30);
        expect(g!.watch.split(/\s+/).length).toBeGreaterThan(20);
        const titles = m.views.map((v) => v.title);
        for (const t of g!.views ?? []) expect(titles, `view “${t}”`).toContain(t);
        const text = [g!.what, g!.watch, ...(g!.try ?? []).map((t) => t.note)].join(' ');
        expect(unknownQuotes(m, text), 'quoted names').toEqual([]);
        for (const t of g!.try ?? []) {
          for (const [k, v] of Object.entries(t.patch)) {
            const f = m.configSchema.find((x) => x.key === k);
            expect(f, `setting ${k}`).toBeDefined();
            expect(allowed(f!, v), `${k} = ${String(v)}`).toBe(true);
          }
          expect(m.validate?.({ ...m.defaults, ...t.patch }) ?? null).toBeNull();
        }
      });

      it('applies every suggested change and trains one step with a finite loss', async () => {
        for (const t of m.guide?.try ?? []) {
          const s = new Session(m);
          await s.loadData(fsLoader);
          s.setConfig(t.patch);
          await s.loadData(fsLoader);
          s.inspect({ kind: 'loss' });
          if (t.train && m.training.support !== 'none') {
            const r = s.trainStep() as TrainRecord;
            for (const v of Object.values(r.losses)) expect(Number.isFinite(v), t.label).toBe(true);
          }
        }
      }, 120_000);
    });
  }
});
