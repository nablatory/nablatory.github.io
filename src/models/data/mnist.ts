import { Tensor, type Rng } from '../../engine';
import type { AssetLoader, Batch, Config, Dataset } from '../types';

/**
 * MNIST subsets shipped in public/data/mnist (first 6,000 training and first
 * 1,000 test images of the original files). © Y. LeCun, C. Cortes,
 * C. J. C. Burges, CC BY-SA 3.0.
 */
export interface MnistData {
  trainImages: Uint8Array;
  trainLabels: Uint8Array;
  testImages: Uint8Array;
  testLabels: Uint8Array;
}

export const MNIST_LICENSE =
  'MNIST © LeCun, Cortes & Burges, CC BY-SA 3.0 (subset redistributed unmodified)';
export const DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

let cache: Promise<MnistData> | null = null;
let loaded: MnistData | null = null;

export function loadMnist(loader: AssetLoader): Promise<MnistData> {
  cache ??= Promise.all(
    ['train-images', 'train-labels', 'test-images', 'test-labels'].map((n) =>
      loader(`data/mnist/${n}.u8`).then((b) => new Uint8Array(b)),
    ),
  ).then(([trainImages, trainLabels, testImages, testLabels]) => {
    loaded = { trainImages, trainLabels, testImages, testLabels };
    return loaded;
  });
  return cache;
}

export function mnist(): MnistData {
  if (!loaded) throw new Error('MNIST has not been loaded');
  return loaded;
}

/** Inject data directly (offline training scripts). */
export function setMnist(d: MnistData): void {
  loaded = d;
  cache = Promise.resolve(d);
}

/**
 * Pixel encodings:
 * lecun     – 28×28 centred in a 32×32 field, background −0.1, ink 1.175 (LeCun et al., 1998)
 * unit      – 28×28 in [0, 1]
 * symmetric – 28×28 in [−1, 1]
 * half      – 14×14 (2×2 average) in [−1, 1]
 */
export type MnistEncoding = 'lecun' | 'unit' | 'symmetric' | 'half';

export function encodingShape(enc: MnistEncoding): [number, number] {
  return enc === 'lecun' ? [32, 32] : enc === 'half' ? [14, 14] : [28, 28];
}

export function encodeDigit(
  src: ArrayLike<number>,
  srcOff: number,
  enc: MnistEncoding,
  out: Float32Array,
  outOff: number,
): void {
  if (enc === 'lecun') {
    for (let i = 0; i < 32 * 32; i++) out[outOff + i] = -0.1;
    for (let y = 0; y < 28; y++)
      for (let x = 0; x < 28; x++) {
        out[outOff + (y + 2) * 32 + x + 2] = -0.1 + 1.275 * (src[srcOff + y * 28 + x] / 255);
      }
  } else if (enc === 'half') {
    for (let y = 0; y < 14; y++)
      for (let x = 0; x < 14; x++) {
        const s =
          src[srcOff + 2 * y * 28 + 2 * x] +
          src[srcOff + 2 * y * 28 + 2 * x + 1] +
          src[srcOff + (2 * y + 1) * 28 + 2 * x] +
          src[srcOff + (2 * y + 1) * 28 + 2 * x + 1];
        out[outOff + y * 14 + x] = (s / (4 * 255)) * 2 - 1;
      }
  } else {
    for (let i = 0; i < 784; i++) {
      const v = src[srcOff + i] / 255;
      out[outOff + i] = enc === 'unit' ? v : v * 2 - 1;
    }
  }
}

export function digitBatch(
  images: Uint8Array,
  labels: Uint8Array,
  indices: number[],
  enc: MnistEncoding,
): Batch {
  const [h, w] = encodingShape(enc);
  const x = new Float32Array(indices.length * h * w);
  indices.forEach((idx, k) => encodeDigit(images, idx * 784, enc, x, k * h * w));
  const ls = indices.map((i) => labels[i]);
  return {
    inputs: { image: new Tensor([indices.length, 1, h, w], x) },
    labels: ls,
    meta: {
      classNames: DIGITS,
      caption: indices.length === 1 ? `test digit #${indices[0]} (label ${ls[0]})` : undefined,
    },
  };
}

export const MNIST_PREPROCESSING: Record<MnistEncoding, string> = {
  lecun: 'Centred in a 32×32 field; background −0.1, ink 1.175 (LeCun et al., 1998).',
  unit: 'Pixel intensity / 255, in [0, 1].',
  symmetric: '2 · intensity / 255 − 1, in [−1, 1].',
  half: '2×2 average to 14×14, then mapped to [−1, 1].',
};

export function mnistDataset<C extends Config>(
  encoding: (config: C) => MnistEncoding,
  opts: { validationSize?: number } = {},
): Dataset<C> {
  const valN = opts.validationSize ?? 500;
  return {
    id: 'mnist',
    name: 'MNIST digits',
    description: 'Handwritten digits, 6,000 training / 1,000 test images (subset of MNIST).',
    source: 'static',
    license: MNIST_LICENSE,
    load: async (loader) => {
      await loadMnist(loader);
    },
    train(rng: Rng, batchSize: number, config: C) {
      const d = mnist();
      const n = d.trainLabels.length;
      const idx = Array.from({ length: batchSize }, () => rng.int(n));
      return digitBatch(d.trainImages, d.trainLabels, idx, encoding(config));
    },
    validation(config: C) {
      const d = mnist();
      const idx = Array.from({ length: Math.min(valN, d.testLabels.length) }, (_, i) => i);
      return digitBatch(d.testImages, d.testLabels, idx, encoding(config));
    },
    exampleCount: 24,
    example(index: number, config: C) {
      const d = mnist();
      return digitBatch(d.testImages, d.testLabels, [index], encoding(config));
    },
    exampleLabel(index: number) {
      const d = loaded;
      return d ? `Test #${index} · ${d.testLabels[index]}` : `Test #${index}`;
    },
  };
}
