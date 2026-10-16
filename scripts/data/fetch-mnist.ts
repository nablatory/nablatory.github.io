/**
 * Downloads MNIST (LeCun, Cortes & Burges; CC BY-SA 3.0) into scripts/.data
 * and writes the subsets shipped with the site to public/data/mnist:
 *   train-images.u8 / train-labels.u8   first 6,000 training images
 *   test-images.u8  / test-labels.u8    first 1,000 test images
 * Images are raw uint8, 28×28, row-major; labels are uint8.
 *
 * Usage: npx tsx scripts/data/fetch-mnist.ts
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { join } from 'node:path';

const MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist';
const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
};
export const CACHE = join(import.meta.dirname, '..', '.data', 'mnist');
const OUT = join(import.meta.dirname, '..', '..', 'public', 'data', 'mnist');

async function fetchFile(name: string): Promise<Uint8Array> {
  const path = join(CACHE, name);
  if (!existsSync(path)) {
    const res = await fetch(`${MIRROR}/${name}`);
    if (!res.ok) throw new Error(`download ${name}: HTTP ${res.status}`);
    writeFileSync(path, new Uint8Array(await res.arrayBuffer()));
  }
  return new Uint8Array(gunzipSync(readFileSync(path)));
}

function parseIdx(buf: Uint8Array): { dims: number[]; data: Uint8Array } {
  const view = new DataView(buf.buffer, buf.byteOffset);
  const magic = view.getUint32(0);
  const nd = magic & 0xff;
  const dims = Array.from({ length: nd }, (_, i) => view.getUint32(4 + 4 * i));
  return { dims, data: buf.subarray(4 + 4 * nd) };
}

export async function loadMnist() {
  mkdirSync(CACHE, { recursive: true });
  const [ti, tl, vi, vl] = await Promise.all(Object.values(FILES).map(fetchFile));
  return {
    trainImages: parseIdx(ti).data,
    trainLabels: parseIdx(tl).data,
    testImages: parseIdx(vi).data,
    testLabels: parseIdx(vl).data,
  };
}

async function main() {
  const m = await loadMnist();
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, 'train-images.u8'), m.trainImages.subarray(0, 6000 * 784));
  writeFileSync(join(OUT, 'train-labels.u8'), m.trainLabels.subarray(0, 6000));
  writeFileSync(join(OUT, 'test-images.u8'), m.testImages.subarray(0, 1000 * 784));
  writeFileSync(join(OUT, 'test-labels.u8'), m.testLabels.subarray(0, 1000));
  console.log(
    `train ${m.trainLabels.length}, test ${m.testLabels.length}; subsets written to ${OUT}`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
