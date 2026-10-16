import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type { AssetLoader } from '../src/models/types';

export const PUBLIC = join(import.meta.dirname, '..', 'public');

export const fsLoader: AssetLoader = async (path) => {
  const full = join(PUBLIC, path);
  if (!existsSync(full)) throw new Error(`asset not found: ${path}`);
  const b = readFileSync(full);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};
