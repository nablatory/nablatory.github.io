/**
 * Trains U-Net (default configuration, with 2 or 3 classes) on the synthetic
 * shapes dataset and writes public/weights/unet-shapes<classes>.{json,bin}.
 * Training scenes come from the session's training stream; test metrics use
 * 512 scenes drawn from a separately seeded generator.
 *
 * Usage: npx tsx scripts/train/unet.ts [classes=2] [steps=1800]
 */
import unet from '../../src/models/convolutional/unet';
import { Session } from '../../src/runtime/session';
import { renderScene, sceneBatch, type ShapesSpec } from '../../src/models/data/shapes';
import { Graph, Rng } from '../../src/engine';
import { saveCheckpoint, log } from './common';

const classes = process.argv[2] === '3' ? '3' : '2';
const steps = Number(process.argv[3] ?? 1800);
const batchSize = 8;
const lr = 3e-3;
const session = new Session(unet, { seed: 1, config: { classes } });
session.batchSize = batchSize;
session.setOptimizer({ kind: 'adam', lr });
const decayAt = Math.floor((2 * steps) / 3);
const t0 = Date.now();
for (let s = 1; s <= steps; s++) {
  if (s === decayAt + 1) session.setOptimizer({ lr: lr * 0.3 });
  const r = session.trainStep() as {
    losses: Record<string, number>;
    metrics: Record<string, number>;
  };
  log(
    s,
    steps,
    `loss ${r.losses.main.toFixed(4)} IoU ${r.metrics.iou.toFixed(3)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`,
  );
}

const c = session.config;
const spec: ShapesSpec = {
  size: Number(c.tile),
  crop: Number(c.tile),
  classes: c.classes === '2' ? 2 : 3,
  noise: Number(c.noise),
};
const testRng = new Rng(20250101).fork('unet-test');
let lossSum = 0;
let iouSum = 0;
let accSum = 0;
const chunks = 16;
for (let k = 0; k < chunks; k++) {
  const batch = sceneBatch(
    Array.from({ length: 32 }, () => renderScene(testRng, spec)),
    spec,
  );
  const g = new Graph({ params: session.params, record: false, grad: false, rng: new Rng(0) });
  const res = g.run(() => unet.forward(batch, c, { training: false, rng: new Rng(0) }));
  lossSum += res.loss!.value.data[0];
  iouSum += res.metrics!.iou;
  accSum += res.metrics!.pixelAccuracy;
}
const v = session.evaluate()!;
const metrics = {
  testLoss: lossSum / chunks,
  testIou: iouSum / chunks,
  testPixelAccuracy: accSum / chunks,
  validationLoss: v.loss,
  validationIou: v.metrics.iou,
  validationPixelAccuracy: v.metrics.pixelAccuracy,
};
console.log(metrics, `${((Date.now() - t0) / 1000).toFixed(0)} s`);

saveCheckpoint(session, `unet-shapes${classes}`, {
  dataset: `Synthetic shapes (src/models/data/shapes.ts), 32×32, ${classes} classes, noise σ = 0.1`,
  script: 'scripts/train/unet.ts',
  metrics,
  notes: `Adam, batch ${batchSize}, ${steps} steps; lr ${lr} then ${lr * 0.3} for the final third. Test metrics: mean over 16 batches of 32 scenes from a separate seed (IoU is computed per batch over the shape classes, then averaged); validation metrics on the browser’s fixed 32-scene validation batch.`,
});
