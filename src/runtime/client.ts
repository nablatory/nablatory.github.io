import type { Envelope, Progress, Reply, Request, ResponseData } from './protocol';

type Pending = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  onProgress?: (p: Progress) => void;
};

/**
 * Promise wrapper around the computation worker. All model arithmetic runs
 * in the worker so the interface stays responsive.
 */
export class EngineClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private listeners = new Set<() => void>();
  /** Requests currently running. */
  busy = 0;

  constructor() {
    this.worker = this.spawn();
  }

  private spawn(): Worker {
    const w = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module',
      name: 'nablatory-engine',
    });
    w.onmessage = (e: MessageEvent<Reply>) => {
      const r = e.data;
      const p = this.pending.get(r.id);
      if (!p) return;
      if (r.kind === 'progress') {
        p.onProgress?.(r.data);
        return;
      }
      this.pending.delete(r.id);
      this.busy--;
      this.emit();
      if (r.kind === 'result') p.resolve(r.data);
      else p.reject(new Error(r.message));
    };
    w.onerror = (e) => {
      const err = new Error(e.message || 'computation worker failed');
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      this.busy = 0;
      this.emit();
    };
    return w;
  }

  request<T extends Request['type']>(
    req: Extract<Request, { type: T }>,
    onProgress?: (p: Progress) => void,
  ): Promise<ResponseData[T]> {
    const id = this.nextId++;
    this.busy++;
    this.emit();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
      const env = { ...req, id } as Envelope;
      // Request payloads are copied (structured clone), never transferred, so
      // callers keep their buffers.
      this.worker.postMessage(env);
    });
  }

  /** Cooperative stop for training loops and procedures. */
  stop(session: string): void {
    const id = this.nextId++;
    this.worker.postMessage({ id, type: 'stop', session } satisfies Envelope);
  }

  /**
   * Terminate the worker (e.g. a computation that does not yield) and start a
   * fresh one. All sessions are lost; callers re-open them.
   */
  restart(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('computation cancelled'));
    this.pending.clear();
    this.busy = 0;
    this.worker = this.spawn();
    this.emit();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    for (const l of this.listeners) l();
  }
}

export const engine = new EngineClient();
