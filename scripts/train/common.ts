import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import type { Session } from '../../src/runtime/session';
import type { CheckpointFile } from '../../src/runtime/checkpoint';

export const WEIGHTS = join(import.meta.dirname, '..', '..', 'public', 'weights');

function gitRevision(): string {
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString()
      .trim();
  } catch {
    return 'unknown';
  }
}

/** Write <name>.json (manifest + provenance) and <name>.bin (float32 little-endian). */
export function saveCheckpoint(
  session: Session,
  name: string,
  provenance: Omit<
    CheckpointFile['provenance'],
    'engineRevision' | 'date' | 'seed' | 'steps' | 'optimizer'
  >,
): void {
  mkdirSync(WEIGHTS, { recursive: true });
  const { manifest, data } = session.params.serialize();
  const file: CheckpointFile = {
    format: 'nablatory-checkpoint',
    version: 1,
    model: session.model.id,
    config: session.config,
    params: manifest,
    provenance: {
      ...provenance,
      seed: session.seed,
      steps: session.step,
      optimizer: session.optimizerConfig,
      engineRevision: gitRevision(),
      date: new Date().toISOString().slice(0, 10),
    },
  };
  writeFileSync(join(WEIGHTS, `${name}.json`), JSON.stringify(file, null, 1));
  writeFileSync(join(WEIGHTS, `${name}.bin`), new Uint8Array(data.buffer));
  console.log(`saved ${name}: ${data.length} floats`);
}

export function log(step: number, total: number, msg: string): void {
  if (step % 50 === 0 || step === total) console.log(`[${step}/${total}] ${msg}`);
}
