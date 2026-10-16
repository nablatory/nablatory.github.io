import type { OptimizerConfig } from '../engine';
import type { AssetLoader, Config } from '../models/types';

export interface CheckpointFile {
  format: 'nablatory-checkpoint';
  version: 1;
  model: string;
  config: Config;
  params: { name: string; shape: number[]; offset: number }[];
  provenance: {
    dataset: string;
    script: string;
    seed: number;
    steps: number;
    optimizer: OptimizerConfig;
    metrics: Record<string, number>;
    engineRevision: string;
    date: string;
    notes?: string;
  };
}

export async function fetchCheckpoint(
  loader: AssetLoader,
  path: string,
): Promise<{ file: CheckpointFile; data: Float32Array }> {
  const [json, bin] = await Promise.all([loader(`${path}.json`), loader(`${path}.bin`)]);
  const file = JSON.parse(new TextDecoder().decode(json)) as CheckpointFile;
  if (file.format !== 'nablatory-checkpoint' || file.version !== 1) {
    throw new Error(`${path}: unsupported checkpoint format`);
  }
  const data = new Float32Array(bin);
  const expected = file.params.reduce((n, p) => n + p.shape.reduce((a, b) => a * b, 1), 0);
  if (data.length !== expected)
    throw new Error(`${path}: ${data.length} values, manifest expects ${expected}`);
  return { file, data };
}
