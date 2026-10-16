/**
 * Trains ResNet (default configuration) on the full MNIST training set and
 * writes public/weights/resnet-mnist.{json,bin}.
 *
 * Usage: npx tsx scripts/train/resnet.ts [steps=1800]
 */
import resnet from '../../src/models/convolutional/resnet';
import { trainMnistClassifier } from './vgg-resnet-mnist';

await trainMnistClassifier(resnet, 'resnet-mnist', {
  steps: Number(process.argv[2] ?? 1800),
  batch: 32,
  lr: 5e-3,
});
