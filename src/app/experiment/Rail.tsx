import { useState, type ReactNode } from 'react';
import type { ModelDef, ConfigField, Config } from '../../models/types';
import type { OptimizerKind, ClipMode } from '../../engine/optim';
import { useStore } from '../../state/store';
import type { Experiment } from './controller';
import { Button, Field, NumberInput, Select, TextInput, Toggle } from '../../ui/controls';
import { Term } from '../../ui/Term';
import { DrawPad } from './DrawPad';
import { findTerm } from '../../ui/glossary';

export function ConfigFields({
  fields,
  config,
  onChange,
  disabled,
  inline,
}: {
  fields: ConfigField[];
  config: Config;
  onChange: (key: string, v: number | string | boolean) => void;
  disabled?: boolean;
  /** Label and control on one line, fields side by side (procedure toolbars). */
  inline?: boolean;
}) {
  return (
    <div className={inline ? 'fields fields-inline' : 'fields'}>
      {fields.map((f) => (
        <Field key={f.key} label={f.label} hint={f.help} term={findTerm(f.label)} inline={inline}>
          {f.kind === 'select' ? (
            <Select
              value={String(config[f.key])}
              options={f.options}
              onChange={(v) => onChange(f.key, v)}
              disabled={disabled}
              ariaLabel={f.label}
            />
          ) : f.kind === 'text' ? (
            <TextInput
              value={String(config[f.key] ?? '')}
              onCommit={(v) => onChange(f.key, v)}
              validate={f.validate}
              maxLength={f.maxLength}
              suggestions={f.suggestions}
              disabled={disabled}
              ariaLabel={f.label}
            />
          ) : f.kind === 'bool' ? (
            <Toggle
              checked={!!config[f.key]}
              onChange={(v) => onChange(f.key, v)}
              label={config[f.key] ? 'On' : 'Off'}
              disabled={disabled}
            />
          ) : (
            <NumberInput
              value={Number(config[f.key])}
              min={f.min}
              max={f.max}
              step={f.step ?? (f.kind === 'int' ? 1 : 0.1)}
              log={f.log}
              integer={f.kind === 'int'}
              onCommit={(v) => onChange(f.key, v)}
              disabled={disabled}
              ariaLabel={f.label}
            />
          )}
        </Field>
      ))}
    </div>
  );
}

/** One collapsible group of the settings rail. */
function RailSection({
  title,
  meta,
  desc,
  defaultOpen,
  children,
}: {
  title: string;
  meta?: ReactNode;
  /** One line under the heading: what this group changes. */
  desc?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <details className="rail-section" open={defaultOpen || undefined}>
      <summary>
        <svg viewBox="0 0 8 8" aria-hidden="true">
          <path d="M2 1l3.5 3L2 7" fill="none" stroke="currentColor" strokeWidth="1.3" />
        </svg>
        <span className="rail-h">{title}</span>
        {meta && <span className="rail-meta">{meta}</span>}
      </summary>
      <div className="rail-body">
        {desc && <p className="rail-desc">{desc}</p>}
        {children}
      </div>
    </details>
  );
}

function SchemaSection({
  exp,
  model,
  affects,
  title,
  desc,
  defaultOpen,
}: {
  exp: Experiment;
  model: ModelDef;
  affects: ConfigField['affects'];
  title: string;
  desc: string;
  defaultOpen?: boolean;
}) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const fields = model.configSchema.filter((f) => f.affects === affects);
  if (!summary || !fields.length) return null;
  return (
    <RailSection title={title} desc={desc} defaultOpen={defaultOpen}>
      <ConfigFields
        fields={fields}
        config={summary.config}
        disabled={!!busy}
        onChange={(k, v) => exp.setConfig({ [k]: v })}
      />
    </RailSection>
  );
}

/** Settings that change the network itself (and so its weights). */
export function ConfigPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  return (
    <SchemaSection
      exp={exp}
      model={model}
      affects="architecture"
      title="Architecture"
      desc="Layers, sizes and connections. A change rebuilds the network from random weights."
      defaultOpen
    />
  );
}

/** Settings that change the forward computation but keep the weights. */
export function ComputationPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  return (
    <SchemaSection
      exp={exp}
      model={model}
      affects="forward"
      title="Computation"
      desc="Switches in the forward pass and the loss. The weights stay as they are."
      defaultOpen
    />
  );
}

/** Settings of the data the model trains and is evaluated on. */
export function DataPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  return (
    <SchemaSection
      exp={exp}
      model={model}
      affects="data"
      title="Data"
      desc="What the model is trained and evaluated on."
    />
  );
}

/** Example picker for the action bar, with an optional drawing pad. */
export function InputPicker({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const [drawing, setDrawing] = useState(false);
  if (!summary) return null;
  const options = summary.exampleLabels.map((label, i) => ({ value: String(i), label }));
  if (summary.example === 'custom') options.push({ value: 'custom', label: 'Your input' });
  return (
    <div className="bar-group bar-input">
      <span className="bar-label">Input</span>
      <Select
        value={String(summary.example)}
        options={options}
        onChange={(v) => v !== 'custom' && exp.setExample(Number(v))}
        disabled={!!busy}
        ariaLabel="Input example"
      />
      {model.customInput && (
        <Button size="sm" aria-expanded={drawing} onClick={() => setDrawing(!drawing)}>
          {drawing ? 'Close' : 'Draw'}
        </Button>
      )}
      {drawing && model.customInput && (
        <div className="bar-pop">
          <DrawPad
            spec={model.customInput}
            onUse={(t) => {
              exp.setCustomInput(t);
              setDrawing(false);
            }}
          />
        </div>
      )}
    </div>
  );
}

/** Step / Train / Pause / Reset for the action bar. */
export function TrainControls({
  exp,
  model,
  onTrain,
  onStep,
  onReset,
}: {
  exp: Experiment;
  model: ModelDef;
  /** Replaces the default single training request (the Guide trains in visible chunks). */
  onTrain?: (steps: number) => void;
  onStep?: () => void;
  onReset?: () => void;
}) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const progress = useStore(exp.store, (s) => s.progress);
  const live = useStore(exp.store, (s) => s.live);
  const [steps, setSteps] = useState(model.training.steps);
  if (!summary) return null;
  if (model.training.support === 'none') {
    return <span className="bar-group muted">Inference only</span>;
  }
  const training = !!live || !!busy?.startsWith('Training');
  const frac = live ? (live.total ? live.done / live.total : null) : progress;
  return (
    <div className="bar-group bar-train">
      {training ? (
        <Button variant="primary" onClick={() => exp.stop()}>
          Pause
        </Button>
      ) : (
        <Button
          variant="primary"
          disabled={!!busy}
          onClick={() => (onTrain ? onTrain(steps) : exp.train(steps))}
          title={`Run ${steps} optimisation steps`}
        >
          Train
        </Button>
      )}
      <NumberInput
        value={steps}
        min={1}
        max={20000}
        integer
        onCommit={setSteps}
        width={68}
        ariaLabel="Steps to train"
      />
      <span className="bar-unit">steps</span>
      <Button
        disabled={!!busy || training}
        onClick={() => (onStep ? onStep() : exp.step())}
        title="One optimisation step: forward, loss, backward, update"
      >
        Step
      </Button>
      {onReset && (
        <Button
          variant="quiet"
          disabled={!!busy || training}
          onClick={onReset}
          title="Back to the default settings and the weights this page opened with"
        >
          Reset
        </Button>
      )}
      {training && frac !== null && (
        <span
          className="bar-progress"
          role="progressbar"
          aria-label="Training progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(frac * 100)}
        >
          <span style={{ width: `${frac * 100}%` }} />
        </span>
      )}
    </div>
  );
}

export function WeightsPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const [seed, setSeed] = useState(summary?.seed ?? 1);
  if (!summary) return null;
  const current = summary.lineage[summary.lineage.length - 1];
  return (
    <RailSection
      title="Weights"
      meta={summary.checkpoint ? 'Trained' : `Seed ${summary.seed}`}
      desc="Where the current parameters came from. Load the trained ones or draw new random ones."
    >
      {current && <p className="rail-note">{current}</p>}
      {model.checkpoints?.map((c) => (
        <Button
          key={c.id}
          size="sm"
          disabled={!!busy}
          onClick={() => exp.loadCheckpoint(c.id)}
          title={c.description}
        >
          {model.checkpoints!.length === 1 ? 'Load trained weights' : `Load “${c.label}”`}
        </Button>
      ))}
      <div className="row">
        <NumberInput
          value={seed}
          integer
          min={0}
          max={2 ** 31 - 1}
          onCommit={setSeed}
          width={84}
          ariaLabel="Seed"
        />
        <Button size="sm" disabled={!!busy} onClick={() => exp.reinit(seed)}>
          Re-initialise
        </Button>
      </div>
      {summary.edits.length > 0 && (
        <div className="row">
          <span className="muted">{summary.edits.length} edited</span>
          <Button size="sm" variant="quiet" onClick={() => exp.revertEdits()}>
            Revert
          </Button>
        </div>
      )}
      {summary.overrides.map((o) => (
        <div key={o.key} className="row">
          <span className="muted" title={o.key}>
            Patched {o.key.split('/').pop()}
          </span>
          <Button size="sm" variant="quiet" onClick={() => exp.setOverride(o.key, [])}>
            Remove
          </Button>
        </div>
      ))}
    </RailSection>
  );
}

const OPTIMIZERS: { value: OptimizerKind; label: string }[] = [
  { value: 'sgd', label: 'SGD' },
  { value: 'momentum', label: 'SGD + momentum' },
  { value: 'nesterov', label: 'Nesterov' },
  { value: 'rmsprop', label: 'RMSProp' },
  { value: 'adam', label: 'Adam' },
  { value: 'adamw', label: 'AdamW' },
];

export function OptimizerPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  if (!summary || model.training.support === 'none') return null;
  const o = summary.optimizer;
  const set = (patch: Partial<typeof o>) => exp.setOptimizer(patch);
  // RMSProp's smoothing constant and Adam's β₂ share a field; each optimiser starts from its own
  // default (PyTorch: α = 0.99, β₂ = 0.999, or the model's own setting for its optimiser).
  const setKind = (kind: OptimizerKind) => {
    const fam = (k: OptimizerKind) =>
      k === 'rmsprop' ? 'r' : k === 'adam' || k === 'adamw' ? 'a' : 's';
    if (fam(kind) === fam(o.kind) || fam(kind) === 's') return set({ kind });
    const home = model.training.optimizer;
    const beta2 =
      kind === 'rmsprop'
        ? home.kind === 'rmsprop'
          ? home.beta2
          : 0.99
        : home.kind === 'adam' || home.kind === 'adamw'
          ? home.beta2
          : 0.999;
    set({ kind, beta2 });
  };
  const groups = groupParams(summary.params.filter((p) => p.trainable).map((p) => p.name));
  const frozen = new Set(summary.params.filter((p) => p.frozen).map((p) => p.name));
  const label = OPTIMIZERS.find((x) => x.value === o.kind)?.label ?? o.kind;
  return (
    <RailSection
      title="Optimiser"
      meta={label}
      desc="How each training step turns gradients into weight updates."
    >
      <div className="fields">
        <Field label="Optimiser" term="optimiser">
          <Select value={o.kind} options={OPTIMIZERS} onChange={setKind} ariaLabel="Optimiser" />
        </Field>
        <div className="fields-2">
          <Field label="Learning rate η" term="lr">
            <NumberInput
              value={o.lr}
              min={1e-7}
              max={10}
              log
              onCommit={(lr) => set({ lr })}
              ariaLabel="Learning rate"
            />
          </Field>
          <Field label="Batch size" term="batch">
            <NumberInput
              value={summary.batchSize}
              min={1}
              max={512}
              integer
              onCommit={(b) => exp.setOptimizer({}, b)}
              ariaLabel="Batch size"
            />
          </Field>
          {(o.kind === 'momentum' || o.kind === 'nesterov') && (
            <Field label="Momentum μ" term="momentum">
              <NumberInput
                value={o.momentum}
                min={0}
                max={0.999}
                step={0.01}
                onCommit={(momentum) => set({ momentum })}
                ariaLabel="Momentum"
              />
            </Field>
          )}
          {(o.kind === 'adam' || o.kind === 'adamw') && (
            <Field label="β₁" term="beta1">
              <NumberInput
                value={o.beta1}
                min={0}
                max={0.999}
                step={0.01}
                onCommit={(beta1) => set({ beta1 })}
                ariaLabel="beta 1"
              />
            </Field>
          )}
          {(o.kind === 'adam' || o.kind === 'adamw' || o.kind === 'rmsprop') && (
            <Field label={o.kind === 'rmsprop' ? 'Smoothing α' : 'β₂'} term="beta2">
              <NumberInput
                value={o.beta2}
                min={0}
                max={0.9999}
                step={0.001}
                onCommit={(beta2) => set({ beta2 })}
                ariaLabel="beta 2"
              />
            </Field>
          )}
          <Field
            label="Weight decay"
            term="weightDecay"
            hint={o.kind === 'adamw' ? 'Decoupled (AdamW).' : 'L2 penalty added to the gradient.'}
          >
            <NumberInput
              value={o.weightDecay}
              min={0}
              max={1}
              log={o.weightDecay > 0}
              step={1e-4}
              onCommit={(weightDecay) => set({ weightDecay })}
              ariaLabel="Weight decay"
            />
          </Field>
        </div>
        <div className="fields-2">
          <Field label="Clipping" term="clip">
            <Select<ClipMode>
              value={o.clip}
              options={[
                { value: 'none', label: 'None' },
                { value: 'norm', label: 'Global norm' },
                { value: 'value', label: 'Per value' },
              ]}
              onChange={(clip) => set({ clip })}
              ariaLabel="Gradient clipping"
            />
          </Field>
          {o.clip !== 'none' && (
            <Field label={o.clip === 'norm' ? 'Max ‖g‖' : 'Max |g|'} term="clip">
              <NumberInput
                value={o.clipValue}
                min={1e-4}
                max={1e4}
                log
                onCommit={(clipValue) => set({ clipValue })}
                ariaLabel="Clip value"
              />
            </Field>
          )}
        </div>
      </div>
      {groups.length > 1 && (
        <div className="rail-group">
          <h3 className="rail-sub">
            <Term id="freeze">Trainable layers</Term>
            {frozen.size > 0 && <span className="rail-sub-note">{frozen.size} frozen</span>}
          </h3>
          <div className="toggle-list">
            {groups.map(([g, names]) => {
              const isFrozen = names.every((n) => frozen.has(n));
              return (
                <Toggle
                  key={g}
                  checked={!isFrozen}
                  label={g}
                  onChange={(on) => {
                    const next = new Set(frozen);
                    for (const n of names) {
                      if (on) next.delete(n);
                      else next.add(n);
                    }
                    exp.setFrozen([...next]);
                  }}
                />
              );
            })}
          </div>
        </div>
      )}
    </RailSection>
  );
}

/** Group parameter names by their leading component (layer). */
function groupParams(names: string[]): [string, string[]][] {
  const m = new Map<string, string[]>();
  for (const n of names) {
    const g = n.includes('.') ? n.slice(0, n.lastIndexOf('.')) : n;
    if (!m.has(g)) m.set(g, []);
    m.get(g)!.push(n);
  }
  return [...m];
}

export { RailSection };
