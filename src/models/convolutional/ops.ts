import { Tensor, alloc, op, type Var } from '../../engine';

/**
 * Strided subsampling of an NCHW tensor: y[n, c, i, j] = x[n, c, s·i, s·j].
 * Output size ⌈H/s⌉ × ⌈W/s⌉, which equals the output size of a 3×3,
 * padding-1 convolution with the same stride. Used by the parameter-free
 * ResNet shortcut (option A). The VJP scatters ∂L/∂y back to the sampled
 * positions; every other input position receives zero.
 */
export function subsample2d(x: Var, stride: number): Var {
  const [N, C, H, W] = x.shape;
  const OH = Math.ceil(H / stride);
  const OW = Math.ceil(W / stride);
  const xd = x.value.data;
  const out = alloc(N * C * OH * OW, xd);
  for (let nc = 0; nc < N * C; nc++)
    for (let i = 0; i < OH; i++)
      for (let j = 0; j < OW; j++)
        out[(nc * OH + i) * OW + j] = xd[(nc * H + i * stride) * W + j * stride];
  return op(
    'subsample2d',
    new Tensor([N, C, OH, OW], out),
    [x],
    (g) => {
      const gx = alloc(xd.length, xd);
      for (let nc = 0; nc < N * C; nc++)
        for (let i = 0; i < OH; i++)
          for (let j = 0; j < OW; j++)
            gx[(nc * H + i * stride) * W + j * stride] = g.data[(nc * OH + i) * OW + j];
      return [new Tensor(x.shape, gx)];
    },
    { stride },
  );
}
