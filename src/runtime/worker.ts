/// <reference lib="webworker" />
import { Tensor } from '../engine';
import { findModel } from '../models/catalog';
import type { AssetLoader } from '../models/types';
import { Session, type StepDetail, type TrainRecord } from './session';
import { fetchCheckpoint } from './checkpoint';
import { traceBuffers, type Trace } from './trace';
import type { Capabilities, Envelope, Inspected, Progress, Reply } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

const sessions = new Map<string, Session>();
const stopRequested = new Set<string>();
const base = import.meta.env.BASE_URL;

const loader: AssetLoader = async (path) => {
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`could not load ${path} (HTTP ${res.status})`);
  return res.arrayBuffer();
};

function get(id: string): Session {
  const s = sessions.get(id);
  if (!s) throw new Error(`no session ${id}`);
  return s;
}

function inspected(s: Session): Inspected {
  return { summary: s.summary(), trace: s.inspect() };
}

const yieldToEvents = () => new Promise<void>((r) => setTimeout(r, 0));

function progress(id: number, data: Progress) {
  self.postMessage({ id, kind: 'progress', data } satisfies Reply);
}

async function capabilities(): Promise<Capabilities> {
  let adapter: string | null = null;
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (gpu) {
    try {
      const a = await gpu.requestAdapter();
      if (a)
        adapter = [a.info?.vendor, a.info?.architecture].filter(Boolean).join(' ') || 'available';
    } catch {
      adapter = null;
    }
  }
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    webgpu: !!adapter,
    adapter,
    cores: nav.hardwareConcurrency ?? 1,
    memoryGB: nav.deviceMemory ?? null,
  };
}

async function handle(msg: Envelope): Promise<unknown> {
  switch (msg.type) {
    case 'capabilities':
      return capabilities();
    case 'open': {
      const model = findModel(msg.model);
      if (!model) throw new Error(`unknown model ${msg.model}`);
      const s = new Session(model, { config: msg.config, seed: msg.seed });
      progress(msg.id, { type: 'load', label: `Loading ${s.dataset.name}` });
      await s.loadData(loader);
      const ck = msg.checkpoint === undefined ? model.checkpoints?.[0]?.id : msg.checkpoint;
      let provenance: unknown;
      if (ck) {
        const meta = model.checkpoints?.find((c) => c.id === ck);
        if (meta) {
          progress(msg.id, { type: 'load', label: `Loading ${meta.label}` });
          const { file, data } = await fetchCheckpoint(loader, meta.path);
          // A checkpoint fixes the architecture it was trained with.
          s.setConfig(file.config);
          s.loadCheckpoint(ck, file.params, data);
          provenance = file.provenance;
        }
      }
      sessions.set(msg.session, s);
      return { ...inspected(s), provenance };
    }
    case 'close':
      sessions.delete(msg.session);
      return null;
    case 'inspect': {
      const s = get(msg.session);
      return { summary: s.summary(), trace: s.inspect(msg.target) };
    }
    case 'setConfig': {
      const s = get(msg.session);
      const datasetBefore = s.dataset;
      const { reinitialized } = s.setConfig(msg.patch);
      if (s.dataset !== datasetBefore) await s.loadData(loader);
      return { ...inspected(s), reinitialized };
    }
    case 'setExample': {
      const s = get(msg.session);
      s.example = msg.index;
      return inspected(s);
    }
    case 'setCustomInput': {
      const s = get(msg.session);
      s.setCustomInput(new Tensor(msg.input.shape, msg.input.data));
      return inspected(s);
    }
    case 'editParam': {
      const s = get(msg.session);
      s.editParam(msg.param, msg.index, msg.value);
      return inspected(s);
    }
    case 'revertEdits': {
      const s = get(msg.session);
      s.revertEdits();
      return inspected(s);
    }
    case 'setOverride': {
      const s = get(msg.session);
      s.setOverride(msg.key, msg.edits);
      return inspected(s);
    }
    case 'setOptimizer': {
      const s = get(msg.session);
      s.setOptimizer(msg.optimizer, msg.batchSize);
      if (msg.resetState) s.resetOptimizerState();
      return { summary: s.summary() };
    }
    case 'setFrozen': {
      const s = get(msg.session);
      s.frozen = new Set(msg.names);
      return inspected(s);
    }
    case 'reinit': {
      const s = get(msg.session);
      s.reinitialize(msg.seed);
      return inspected(s);
    }
    case 'loadCheckpoint': {
      const s = get(msg.session);
      const meta = s.model.checkpoints?.find((c) => c.id === msg.checkpoint);
      if (!meta) throw new Error(`unknown checkpoint ${msg.checkpoint}`);
      const { file, data } = await fetchCheckpoint(loader, meta.path);
      s.setConfig(file.config);
      s.loadCheckpoint(meta.id, file.params, data);
      return { ...inspected(s), provenance: file.provenance };
    }
    case 'step': {
      const s = get(msg.session);
      const detail = s.trainStep({ detail: true }) as StepDetail;
      return { ...inspected(s), detail };
    }
    case 'train': {
      const s = get(msg.session);
      stopRequested.delete(msg.session);
      let stopped = false;
      let lastYield = performance.now();
      for (let i = 0; i < msg.steps; i++) {
        if (stopRequested.has(msg.session)) {
          stopped = true;
          break;
        }
        const record = s.trainStep() as TrainRecord;
        progress(msg.id, { type: 'train', record, done: i + 1, total: msg.steps });
        if (performance.now() - lastYield > 24) {
          await yieldToEvents();
          lastYield = performance.now();
        }
      }
      stopRequested.delete(msg.session);
      return { ...inspected(s), history: s.history, stopped };
    }
    case 'stop':
      stopRequested.add(msg.session);
      return null;
    case 'procedure': {
      const s = get(msg.session);
      stopRequested.delete(msg.session);
      const t0 = performance.now();
      let lastPost = 0;
      const result = s.runProcedure(msg.procedure, msg.args, {
        shouldStop: () => stopRequested.has(msg.session),
        progress: (fraction) => {
          const now = performance.now();
          if (now - lastPost > 50) {
            lastPost = now;
            progress(msg.id, { type: 'procedure', fraction });
          }
        },
      });
      return { result, ms: performance.now() - t0, stopped: stopRequested.delete(msg.session) };
    }
    case 'gradcheck':
      return get(msg.session).gradCheck(msg.names);
    case 'getParam': {
      const t = get(msg.session).params.get(msg.name);
      return { shape: t.shape, data: Float32Array.from(t.data) };
    }
    case 'vjp':
      return get(msg.session).vjp(msg.key);
    case 'loadParams': {
      const s = get(msg.session);
      s.restoreParams(msg.manifest, msg.data, msg.note);
      return inspected(s);
    }
    case 'exportParams':
      return get(msg.session).params.serialize();
  }
}

function transferables(data: unknown): Transferable[] {
  const out: Transferable[] = [];
  const visit = (v: unknown) => {
    if (!v || typeof v !== 'object') return;
    const o = v as Record<string, unknown>;
    if ('nodes' in o && Array.isArray(o.nodes) && 'outputs' in o) {
      out.push(...traceBuffers(o as unknown as Trace));
      return;
    }
    for (const k of ['trace', 'detail', 'before', 'after']) if (k in o) visit(o[k]);
  };
  visit(data);
  return out;
}

async function respond(msg: Envelope) {
  try {
    const data = await handle(msg);
    self.postMessage({ id: msg.id, kind: 'result', data } satisfies Reply, transferables(data));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ id: msg.id, kind: 'error', message } satisfies Reply);
  }
}

/**
 * Commands for one session run in the order they arrive. Training loops yield
 * between steps, so without this a configuration change or re-initialisation
 * could run in the middle of a training run. `stop` and `capabilities` bypass
 * the queue; different sessions still interleave.
 */
const queues = new Map<string, Promise<void>>();

self.onmessage = (e: MessageEvent<Envelope>) => {
  const msg = e.data;
  if (msg.type === 'stop' || !('session' in msg)) {
    void respond(msg);
    return;
  }
  const key = msg.session;
  const next = (queues.get(key) ?? Promise.resolve()).then(() => respond(msg));
  queues.set(key, next);
  void next.then(() => {
    if (queues.get(key) === next) queues.delete(key);
  });
};
