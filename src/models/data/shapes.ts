import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Synthetic segmentation scenes generated in the browser: one to three
 * ellipses and rectangles (both at random orientation) on a background with
 * a random brightness gradient and additive Gaussian noise. Every pixel is
 * labelled background (0), ellipse (1) or rectangle (2); with two classes
 * both shapes are labelled 1. Shapes are drawn brighter or darker than the
 * background with the same contrast distribution, so the two shape classes
 * cannot be told apart from intensity alone, only from their outlines.
 */
export interface ShapesSpec {
  /** Side of the generated (input) image. */
  size: number;
  /** Side of the centred label crop (equal to `size` unless the model crops its output). */
  crop: number;
  classes: 2 | 3;
  /** Standard deviation of the additive pixel noise. */
  noise: number;
}

export const SHAPE_CLASS_NAMES: Record<2 | 3, string[]> = {
  2: ['background', 'shape'],
  3: ['background', 'ellipse', 'rectangle'],
};

export interface Scene {
  image: Float32Array;
  /** Class index per pixel of the centred crop, row-major. */
  labels: Uint8Array;
  shapes: number;
}

export function renderScene(rng: Rng, spec: ShapesSpec): Scene {
  const S = spec.size;
  const base = rng.range(0.25, 0.55);
  const gx = rng.range(-0.15, 0.15);
  const gy = rng.range(-0.15, 0.15);
  const image = new Float32Array(S * S);
  const cls = new Uint8Array(S * S);
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      image[y * S + x] = base + gx * ((x + 0.5) / S - 0.5) + gy * ((y + 0.5) / S - 0.5);
  const count = 1 + rng.int(3);
  for (let k = 0; k < count; k++) {
    const ellipse = rng.uniform() < 0.5;
    const cx = rng.range(0.18, 0.82) * S;
    const cy = rng.range(0.18, 0.82) * S;
    const a = rng.range(0.09, 0.22) * S;
    const b = rng.range(0.09, 0.22) * S;
    const th = rng.range(0, Math.PI);
    const ct = Math.cos(th);
    const st = Math.sin(th);
    const offset = (rng.uniform() < 0.5 ? -1 : 1) * rng.range(0.25, 0.45);
    for (let y = 0; y < S; y++)
      for (let x = 0; x < S; x++) {
        const dx = x + 0.5 - cx;
        const dy = y + 0.5 - cy;
        const u = ct * dx + st * dy;
        const v = -st * dx + ct * dy;
        const inside = ellipse
          ? (u / a) ** 2 + (v / b) ** 2 <= 1
          : Math.abs(u) <= a * 0.85 && Math.abs(v) <= b * 0.85;
        if (!inside) continue;
        const i = y * S + x;
        image[i] = base + gx * ((x + 0.5) / S - 0.5) + gy * ((y + 0.5) / S - 0.5) + offset;
        cls[i] = spec.classes === 2 ? 1 : ellipse ? 1 : 2;
      }
  }
  for (let i = 0; i < S * S; i++) image[i] += spec.noise * rng.normal();
  const C = spec.crop;
  const o = Math.floor((S - C) / 2);
  const labels = new Uint8Array(C * C);
  for (let y = 0; y < C; y++)
    for (let x = 0; x < C; x++) labels[y * C + x] = cls[(y + o) * S + x + o];
  return { image, labels, shapes: count };
}

export function sceneBatch(scenes: Scene[], spec: ShapesSpec, caption?: string): Batch {
  const S = spec.size;
  const C = spec.crop;
  const x = new Float32Array(scenes.length * S * S);
  const m = new Float32Array(scenes.length * C * C);
  scenes.forEach((s, i) => {
    x.set(s.image, i * S * S);
    m.set(s.labels, i * C * C);
  });
  return {
    inputs: { image: new Tensor([scenes.length, 1, S, S], x) },
    targets: { mask: new Tensor([scenes.length, 1, C, C], m) },
    meta: { classNames: SHAPE_CLASS_NAMES[spec.classes], caption },
  };
}

const exampleRng = (i: number) => new Rng(0x5eed).fork(`scene:${i}`);

export function shapesDataset<C extends Config>(
  spec: (c: C) => ShapesSpec,
  opts: { validationSize?: number } = {},
): Dataset<C> {
  const valN = opts.validationSize ?? 32;
  return {
    id: 'shapes',
    name: 'Synthetic shapes',
    description:
      'Ellipses and rectangles at random positions, sizes and orientations on a noisy background with a brightness gradient, with a class label for every pixel. Generated deterministically in the browser.',
    source: 'synthetic',
    train(rng, batchSize, config) {
      const s = spec(config);
      return sceneBatch(
        Array.from({ length: batchSize }, () => renderScene(rng, s)),
        s,
      );
    },
    validation(config) {
      const s = spec(config);
      const rng = new Rng(0xa11da7e);
      return sceneBatch(
        Array.from({ length: valN }, () => renderScene(rng, s)),
        s,
      );
    },
    exampleCount: 24,
    example(index, config) {
      const s = spec(config);
      const scene = renderScene(exampleRng(index), s);
      return sceneBatch(
        [scene],
        s,
        `scene #${index}, ${scene.shapes} shape${scene.shapes > 1 ? 's' : ''}`,
      );
    },
    exampleLabel(index, config) {
      const scene = renderScene(exampleRng(index), spec(config));
      return `Scene #${index} · ${scene.shapes} shape${scene.shapes > 1 ? 's' : ''}`;
    },
  };
}
