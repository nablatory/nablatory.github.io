/**
 * Trains VGG (default configuration) on the full MNIST training set and
 * writes public/weights/vgg-mnist.{json,bin}.
 *
 * Usage: npx tsx scripts/train/vgg.ts [steps=2000]
 */
import vgg from '../../src/models/convolutional/vgg';
import { trainMnistClassifier } from './vgg-resnet-mnist';

await trainMnistClassifier(vgg, 'vgg-mnist', {
  steps: Number(process.argv[2] ?? 2000),
  batch: 32,
  lr: 1e-3,
});
