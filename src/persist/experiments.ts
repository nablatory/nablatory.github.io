import type { OptimizerConfig, ValueEdit } from '../engine';
import type { Config } from '../models/types';
import type { TrainRecord } from '../runtime/session';

/** Versioned, reproducible description of an experiment. */
export interface SavedExperiment {
  format: 'nablatory-experiment';
  version: 1;
  id: string;
  title: string;
  created: string;
  model: string;
  config: Config;
  seed: number;
  checkpoint: string | null;
  optimizer: OptimizerConfig;
  batchSize: number;
  frozen: string[];
  edits: { param: string; index: number; value: number }[];
  overrides: { key: string; edits: ValueEdit[] }[];
  example: number | 'custom';
  step: number;
  history: TrainRecord[];
  /** Parameter values at save time (needed to reproduce trained weights). */
  weights?: { manifest: { name: string; shape: number[]; offset: number }[]; data: Float32Array };
}

const DB = 'nablatory';
const STORE = 'experiments';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const s = db.createObjectStore(STORE, { keyPath: 'id' });
        s.createIndex('model', 'model');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(
  mode: IDBTransactionMode,
  fn: (s: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function saveExperiment(e: SavedExperiment): Promise<IDBValidKey> {
  return tx('readwrite', (s) => s.put(e));
}

export function deleteExperiment(id: string): Promise<undefined> {
  return tx('readwrite', (s) => s.delete(id));
}

export async function listExperiments(model?: string): Promise<SavedExperiment[]> {
  const all = await tx<SavedExperiment[]>('readonly', (s) =>
    model ? s.index('model').getAll(model) : s.getAll(),
  );
  return all.sort((a, b) => b.created.localeCompare(a.created));
}

function toBase64(data: Float32Array): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000)
    s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function fromBase64(b64: string): Float32Array {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function exportJson(e: SavedExperiment): string {
  const { weights, ...rest } = e;
  return JSON.stringify(
    weights
      ? { ...rest, weights: { manifest: weights.manifest, float32Base64: toBase64(weights.data) } }
      : rest,
    null,
    1,
  );
}

export function importJson(text: string): SavedExperiment {
  const o = JSON.parse(text);
  if (o.format !== 'nablatory-experiment' || o.version !== 1)
    throw new Error('Not a Nablatory experiment file (version 1).');
  for (const k of ['model', 'config', 'seed', 'optimizer'])
    if (!(k in o)) throw new Error(`Experiment file is missing “${k}”.`);
  if (o.weights)
    o.weights = { manifest: o.weights.manifest, data: fromBase64(o.weights.float32Base64) };
  o.id = crypto.randomUUID();
  return o as SavedExperiment;
}
