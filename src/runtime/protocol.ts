import type { GradCheckResult, OptimizerConfig, ValueEdit } from '../engine';
import type { Config, ConfigValue, ProcedureResult } from '../models/types';
import type { GradTarget, SessionSummary, StepDetail, TrainRecord } from './session';
import type { Trace, TensorPayload } from './trace';

export type Request =
  | {
      type: 'open';
      session: string;
      model: string;
      config?: Config;
      seed?: number;
      checkpoint?: string | null;
    }
  | { type: 'close'; session: string }
  | { type: 'inspect'; session: string; target?: GradTarget }
  | { type: 'setConfig'; session: string; patch: Record<string, ConfigValue> }
  | { type: 'setExample'; session: string; index: number }
  | { type: 'setCustomInput'; session: string; input: TensorPayload }
  | { type: 'editParam'; session: string; param: string; index: number; value: number }
  | { type: 'revertEdits'; session: string }
  | { type: 'setOverride'; session: string; key: string; edits: ValueEdit[] }
  | {
      type: 'setOptimizer';
      session: string;
      optimizer: Partial<OptimizerConfig>;
      batchSize?: number;
      resetState?: boolean;
    }
  | { type: 'setFrozen'; session: string; names: string[] }
  | { type: 'reinit'; session: string; seed: number }
  | { type: 'loadCheckpoint'; session: string; checkpoint: string }
  | { type: 'step'; session: string }
  | { type: 'train'; session: string; steps: number }
  | { type: 'stop'; session: string }
  | { type: 'procedure'; session: string; procedure: string; args: Config }
  | { type: 'gradcheck'; session: string; names?: string[] }
  | { type: 'getParam'; session: string; name: string }
  | { type: 'vjp'; session: string; key: string }
  | { type: 'exportParams'; session: string }
  | {
      type: 'loadParams';
      session: string;
      manifest: { name: string; shape: number[]; offset: number }[];
      data: Float32Array;
      note: string;
    }
  | { type: 'capabilities' };

export interface Capabilities {
  webgpu: boolean;
  adapter: string | null;
  cores: number;
  memoryGB: number | null;
}

export interface Inspected {
  summary: SessionSummary;
  trace: Trace;
}

export type ResponseData = {
  open: Inspected & { provenance?: unknown };
  close: null;
  inspect: Inspected;
  setConfig: Inspected & { reinitialized: boolean };
  setExample: Inspected;
  setCustomInput: Inspected;
  editParam: Inspected;
  revertEdits: Inspected;
  setOverride: Inspected;
  setOptimizer: { summary: SessionSummary };
  setFrozen: Inspected;
  reinit: Inspected;
  loadCheckpoint: Inspected & { provenance: unknown };
  step: Inspected & { detail: StepDetail };
  train: Inspected & { history: TrainRecord[]; stopped: boolean };
  stop: null;
  procedure: { result: ProcedureResult; ms: number; stopped: boolean };
  gradcheck: GradCheckResult;
  getParam: TensorPayload;
  vjp: VjpResult | null;
  exportParams: {
    manifest: { name: string; shape: number[]; offset: number }[];
    data: Float32Array;
  };
  capabilities: Capabilities;
  loadParams: Inspected;
};

export interface VjpResult {
  upstream: TensorPayload;
  parents: { key: string; contribution: TensorPayload | null; total: TensorPayload | null }[];
}

export type Envelope = { id: number } & Request;

export type Reply =
  | { id: number; kind: 'result'; data: unknown }
  | { id: number; kind: 'error'; message: string }
  | { id: number; kind: 'progress'; data: Progress };

export type Progress =
  | { type: 'train'; record: TrainRecord; done: number; total: number }
  | { type: 'procedure'; fraction: number }
  | { type: 'load'; label: string };
