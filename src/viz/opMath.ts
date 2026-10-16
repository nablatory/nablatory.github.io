import type { TraceNode } from '../runtime/trace';

export interface OpMath {
  /** Forward equation. */
  forward: string;
  /** Vector–Jacobian products, one per input, with ḡ = ∂L/∂y. */
  backward: string[];
  /** Input names in parent order. */
  inputs: string[];
}

const g = '\\bar g';

/** Forward and backward equations of each engine operation, matching src/engine/ops. */
export function opMath(n: TraceNode): OpMath | null {
  const a = n.attrs;
  switch (n.op) {
    case 'linear':
      return {
        forward: a.bias ? 'y = x W^{\\top} + b' : 'y = x W^{\\top}',
        backward: [
          `\\partial L/\\partial x = ${g}\\, W`,
          `\\partial L/\\partial W = ${g}^{\\top} x`,
          ...(a.bias ? [`\\partial L/\\partial b = \\textstyle\\sum_{\\text{rows}} ${g}`] : []),
        ],
        inputs: a.bias ? ['x', 'W', 'b'] : ['x', 'W'],
      };
    case 'matmul':
      return {
        forward: 'C = A B',
        backward: [
          `\\partial L/\\partial A = ${g}\\, B^{\\top}`,
          `\\partial L/\\partial B = A^{\\top} ${g}`,
        ],
        inputs: ['A', 'B'],
      };
    case 'conv2d': {
      const [s] = (a.stride as number[]) ?? [1];
      const [p] = (a.padding as number[]) ?? [0];
      return {
        forward: `y_{o}(i,j) = b_o + \\sum_{c,u,v} w_{o,c,u,v}\\, x_c(${s}i - ${p} + u,\\; ${s}j - ${p} + v)`,
        backward: [
          `\\partial L/\\partial x = \\operatorname{conv}^{\\top}_{w}(${g})\\;\\text{(full correlation with flipped kernels)}`,
          `\\partial L/\\partial w_{o,c,u,v} = \\sum_{i,j} ${g}_o(i,j)\\, x_c(${s}i-${p}+u,\\, ${s}j-${p}+v)`,
          `\\partial L/\\partial b_o = \\sum_{i,j} ${g}_o(i,j)`,
        ],
        inputs: ['x', 'w', 'b'],
      };
    }
    case 'convTranspose2d':
      return {
        forward: 'y = \\operatorname{conv}^{\\top}_{w}(x) + b',
        backward: [
          `\\partial L/\\partial x = \\operatorname{conv}_{w}(${g})`,
          `\\partial L/\\partial w = \\text{correlation of } x \\text{ with } ${g}`,
          `\\partial L/\\partial b = \\sum ${g}`,
        ],
        inputs: ['x', 'w', 'b'],
      };
    case 'add':
      return {
        forward: 'y = a + b',
        backward: [
          `\\partial L/\\partial a = \\Sigma_{\\text{bcast}}\\, ${g}`,
          `\\partial L/\\partial b = \\Sigma_{\\text{bcast}}\\, ${g}`,
        ],
        inputs: ['a', 'b'],
      };
    case 'sub':
      return { forward: 'y = a - b', backward: [`${g}`, `-${g}`], inputs: ['a', 'b'] };
    case 'mul':
      return {
        forward: 'y = a \\odot b',
        backward: [`${g} \\odot b`, `${g} \\odot a`],
        inputs: ['a', 'b'],
      };
    case 'div':
      return {
        forward: 'y = a / b',
        backward: [`${g} / b`, `-${g} \\odot y / b`],
        inputs: ['a', 'b'],
      };
    case 'scale':
      return { forward: `y = ${a.factor}\\,x`, backward: [`${a.factor}\\,${g}`], inputs: ['x'] };
    case 'addScalar':
      return { forward: `y = x + ${a.value}`, backward: [g], inputs: ['x'] };
    case 'axpby':
      return {
        forward: `y = ${a.alpha}\\,x + ${a.beta}\\,z`,
        backward: [`${a.alpha}\\,${g}`, `${a.beta}\\,${g}`],
        inputs: ['x', 'z'],
      };
    case 'relu':
      return {
        forward: 'y = \\max(0, x)',
        backward: [`${g} \\odot \\mathbb{1}[x > 0]`],
        inputs: ['x'],
      };
    case 'leakyRelu':
      return {
        forward: `y = \\max(${a.alpha}x, x)`,
        backward: [`${g} \\odot (\\mathbb{1}[x>0] + ${a.alpha}\\,\\mathbb{1}[x\\le 0])`],
        inputs: ['x'],
      };
    case 'tanh':
      return { forward: 'y = \\tanh x', backward: [`${g} \\odot (1 - y^2)`], inputs: ['x'] };
    case 'scaledTanh':
      return {
        forward: 'y = 1.7159 \\tanh(\\tfrac23 x)',
        backward: [`${g} \\odot \\tfrac23 (1.7159 - y^2/1.7159)`],
        inputs: ['x'],
      };
    case 'sigmoid':
      return { forward: 'y = 1/(1+e^{-x})', backward: [`${g} \\odot y(1-y)`], inputs: ['x'] };
    case 'gelu':
      return {
        forward: 'y = x\\,\\Phi(x)',
        backward: [`${g} \\odot (\\Phi(x) + x\\,\\varphi(x))`],
        inputs: ['x'],
      };
    case 'silu':
      return {
        forward: 'y = x\\,\\sigma(x)',
        backward: [`${g} \\odot \\sigma(x)(1 + x(1-\\sigma(x)))`],
        inputs: ['x'],
      };
    case 'softplus':
      return {
        forward: 'y = \\log(1 + e^{x})',
        backward: [`${g} \\odot \\sigma(x)`],
        inputs: ['x'],
      };
    case 'elu':
      return {
        forward: 'y = x \\text{ if } x>0 \\text{ else } e^x - 1',
        backward: [`${g} \\odot (\\mathbb 1[x>0] + (y+1)\\mathbb 1[x\\le0])`],
        inputs: ['x'],
      };
    case 'exp':
      return { forward: 'y = e^{x}', backward: [`${g} \\odot y`], inputs: ['x'] };
    case 'log':
      return { forward: 'y = \\log x', backward: [`${g} / x`], inputs: ['x'] };
    case 'sqrt':
      return { forward: 'y = \\sqrt{x}', backward: [`${g} / (2y)`], inputs: ['x'] };
    case 'square':
      return { forward: 'y = x^2', backward: [`2x \\odot ${g}`], inputs: ['x'] };
    case 'softmax':
      return {
        forward: 'y_i = e^{x_i - m} / \\textstyle\\sum_j e^{x_j - m},\\; m = \\max_j x_j',
        backward: [`\\partial L/\\partial x = y \\odot (${g} - \\langle ${g}, y \\rangle)`],
        inputs: ['x'],
      };
    case 'logSoftmax':
      return {
        forward: 'y_i = x_i - \\log \\textstyle\\sum_j e^{x_j}',
        backward: [`${g} - \\operatorname{softmax}(x) \\textstyle\\sum_i ${g}_i`],
        inputs: ['x'],
      };
    case 'sum':
      return {
        forward: `y = \\textstyle\\sum_{\\text{axes } ${(a.axes as number[]).join(',')}} x`,
        backward: [`\\text{broadcast}(${g})`],
        inputs: ['x'],
      };
    case 'mean':
      return {
        forward: `y = \\text{mean}_{\\text{axes } ${(a.axes as number[]).join(',')}}\\, x`,
        backward: [`\\text{broadcast}(${g}) / n`],
        inputs: ['x'],
      };
    case 'maxPool2d':
      return {
        forward: `y(i,j) = \\max_{(u,v) \\in \\text{window}} x(${a.stride}i+u, ${a.stride}j+v)`,
        backward: [`${g} \\text{ routed to the arg-max of each window}`],
        inputs: ['x'],
      };
    case 'avgPool2d':
      return {
        forward: `y(i,j) = \\tfrac{1}{${Number(a.kernel) ** 2}} \\sum_{\\text{window}} x`,
        backward: [`${g}/${Number(a.kernel) ** 2} \\text{ spread over each window}`],
        inputs: ['x'],
      };
    case 'layerNorm':
      return {
        forward: 'y = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta',
        backward: [
          `\\partial L/\\partial x = \\tfrac{1}{\\sigma}\\,(\\hat g - \\overline{\\hat g} - \\hat x\\, \\overline{\\hat g \\hat x}),\\; \\hat g = \\gamma \\odot ${g}`,
          `\\partial L/\\partial \\gamma = \\textstyle\\sum ${g} \\odot \\hat x`,
          `\\partial L/\\partial \\beta = \\textstyle\\sum ${g}`,
        ],
        inputs: ['x', 'γ', 'β'],
      };
    case 'rmsNorm':
      return {
        forward: 'y = \\gamma \\odot x / \\sqrt{\\overline{x^2} + \\epsilon}',
        backward: [
          `r(\\hat g - \\hat x\\, \\overline{\\hat g \\hat x})`,
          `\\textstyle\\sum ${g} \\odot \\hat x`,
        ],
        inputs: ['x', 'γ'],
      };
    case 'batchNorm':
      return {
        forward: 'y = \\gamma \\odot (x - \\mu_B)/\\sqrt{\\sigma_B^2 + \\epsilon} + \\beta',
        backward: [
          `\\tfrac{\\gamma}{\\sigma_B}(${g} - \\overline{${g}} - \\hat x\\,\\overline{${g}\\hat x})`,
          `\\textstyle\\sum ${g} \\odot \\hat x`,
          `\\textstyle\\sum ${g}`,
        ],
        inputs: ['x', 'γ', 'β'],
      };
    case 'crossEntropy':
      return {
        forward: 'L = -\\tfrac1N \\textstyle\\sum_n \\log \\operatorname{softmax}(z_n)_{y_n}',
        backward: [
          '\\partial L/\\partial z = (\\operatorname{softmax}(z) - \\operatorname{onehot}(y))/N',
        ],
        inputs: ['z'],
      };
    case 'mse':
      return {
        forward: 'L = \\tfrac1n \\textstyle\\sum (\\hat y - y)^2',
        backward: ['\\partial L/\\partial \\hat y = \\tfrac2n(\\hat y - y)'],
        inputs: ['ŷ', 'y'],
      };
    case 'bceWithLogits':
      return {
        forward: 'L = \\tfrac1n \\textstyle\\sum \\max(z,0) - z t + \\log(1 + e^{-|z|})',
        backward: ['(\\sigma(z) - t)/n'],
        inputs: ['z'],
      };
    case 'maskedFill':
      return {
        forward: `y = x \\text{ where mask} = 0, \\text{ else } ${a.value === '-Infinity' ? '-\\infty' : a.value}`,
        backward: [`${g} \\odot (1 - \\text{mask})`],
        inputs: ['x'],
      };
    case 'gather':
      return {
        forward: 'y_t = E[\\text{id}_t]',
        backward: [`\\partial L/\\partial E[k] = \\textstyle\\sum_{t: \\text{id}_t = k} ${g}_t`],
        inputs: ['E'],
      };
    case 'dropout':
      return {
        forward: `y = x \\odot m / (1 - ${a.rate}),\\; m \\sim \\text{Bernoulli}(1 - ${a.rate})`,
        backward: [`${g} \\odot m / (1-${a.rate})`],
        inputs: ['x'],
      };
    case 'reshape':
    case 'transpose':
    case 'slice':
    case 'concat':
    case 'pad':
    case 'upsampleNearest2d':
      return {
        forward: `y = \\operatorname{${n.op}}(x)`,
        backward: [`\\text{the inverse rearrangement of } ${g}`],
        inputs: ['x'],
      };
    default:
      return null;
  }
}
