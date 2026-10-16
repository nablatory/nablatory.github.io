/**
 * WebGPU matrix multiplication, used to measure what a GPU path would gain
 * and to validate it against the CPU kernel the engine uses. The engine itself
 * runs on the CPU: its tape is synchronous, and reading GPU results back
 * requires awaiting, so GPU kernels are not part of forward or backward passes.
 */

const WGSL = /* wgsl */ `
struct Dims { m: u32, n: u32, k: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

const T: u32 = 16u;
var<workgroup> ta: array<array<f32, 16>, 16>;
var<workgroup> tb: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = g.y;
  let col = g.x;
  var acc = 0.0;
  let tiles = (d.k + T - 1u) / T;
  for (var t = 0u; t < tiles; t++) {
    let ak = t * T + l.x;
    let bk = t * T + l.y;
    ta[l.y][l.x] = select(0.0, a[row * d.k + ak], row < d.m && ak < d.k);
    tb[l.y][l.x] = select(0.0, b[bk * d.n + col], bk < d.k && col < d.n);
    workgroupBarrier();
    for (var i = 0u; i < T; i++) { acc += ta[l.y][i] * tb[i][l.x]; }
    workgroupBarrier();
  }
  if (row < d.m && col < d.n) { c[row * d.n + col] = acc; }
}`;

export interface GpuMatmul {
  adapter: string;
  run(
    a: Float32Array<ArrayBuffer>,
    b: Float32Array<ArrayBuffer>,
    m: number,
    k: number,
    n: number,
  ): Promise<{ c: Float32Array; ms: number }>;
  destroy(): void;
}

export async function createGpuMatmul(): Promise<GpuMatmul | null> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const info = adapter.info;
  return {
    adapter:
      [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ') ||
      'unknown adapter',
    async run(a, b, m, k, n) {
      const t0 = performance.now();
      const buf = (data: Float32Array<ArrayBuffer>, usage: number) => {
        const g = device.createBuffer({
          size: data.byteLength,
          usage: usage | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(g, 0, data);
        return g;
      };
      const ga = buf(a, GPUBufferUsage.STORAGE);
      const gb = buf(b, GPUBufferUsage.STORAGE);
      const gc = device.createBuffer({
        size: m * n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const gd = buf(
        new Float32Array(new Uint32Array([m, n, k, 0]).buffer),
        GPUBufferUsage.UNIFORM,
      );
      const read = device.createBuffer({
        size: m * n * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [ga, gb, gc, gd].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(n / 16), Math.ceil(m / 16));
      pass.end();
      enc.copyBufferToBuffer(gc, 0, read, 0, m * n * 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const c = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      for (const g of [ga, gb, gc, gd, read]) g.destroy();
      return { c, ms: performance.now() - t0 };
    },
    destroy() {
      device.destroy();
    },
  };
}
