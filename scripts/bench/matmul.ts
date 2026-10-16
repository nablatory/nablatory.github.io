/** Measures the CPU matmul kernels: npx tsx scripts/bench/matmul.ts */
import { mmNN, mmNT, mmTN } from '../../src/engine/ops/linalg';

function bench(name: string, f: () => void, macs: number) {
  for (let i = 0; i < 3; i++) f();
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < 600) {
    f();
    n++;
  }
  const ms = (performance.now() - t0) / n;
  console.log(
    `${name.padEnd(28)} ${ms.toFixed(2).padStart(8)} ms  ${(macs / ms / 1e6).toFixed(2)} GMAC/s`,
  );
}

for (const [m, k, n] of [
  [256, 256, 256],
  [512, 64, 128],
  [32, 784, 128],
  [1600, 144, 16],
]) {
  const A = Float32Array.from({ length: m * k }, () => Math.random() - 0.5);
  const B = Float32Array.from({ length: k * n }, () => Math.random() - 0.5);
  const At = Float32Array.from({ length: k * m }, () => Math.random() - 0.5);
  const Bt = Float32Array.from({ length: n * k }, () => Math.random() - 0.5);
  const C = new Float32Array(m * n);
  const macs = m * k * n;
  bench(`NN ${m}x${k}x${n}`, () => mmNN(A, 0, B, 0, C, 0, m, k, n), macs);
  bench(`NT ${m}x${k}x${n}`, () => mmNT(A, 0, Bt, 0, C, 0, m, k, n), macs);
  bench(`TN ${m}x${k}x${n}`, () => mmTN(At, 0, B, 0, C, 0, m, k, n), macs);
}
