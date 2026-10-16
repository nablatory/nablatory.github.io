import type { MouseEvent, ReactNode } from 'react';
import type { Config, ModelDef, Procedure, ViewSpec } from '../../../models/types';
import { Term } from '../../../ui/Term';
import { FAMILIES } from '../../../models/catalog';
import type { Experiment } from '../../experiment/controller';
import { useStore } from '../../../state/store';
import { ConfigFields, InputPicker, TrainControls } from '../../experiment/Rail';
import { Button, Spinner } from '../../../ui/controls';
import { fmt, fmtMs, metricLabel } from '../../../ui/format';
import { href, modelPath, navigate, onLinkClick, rememberView, type ModelView } from '../../router';

export const FIDELITY: Record<ModelDef['fidelity'], { label: string; title: string }> = {
  exact: {
    label: 'Published size',
    title: 'The published architecture at the size given in the paper',
  },
  equations: {
    label: 'Published equations',
    title: 'The paper defines the equations but no fixed size; the size here is a choice',
  },
  reduced: {
    label: 'Reduced scale',
    title:
      'The published equations with fewer layers, channels or tokens so that it runs in a browser',
  },
};

/** Views that apply to the current configuration. */
export function visibleViews(model: ModelDef, config: Config | undefined): ViewSpec[] {
  return model.views.filter((v) => !v.when || (config !== undefined && v.when(config)));
}

/** Busy label of a running procedure; also tells its toolbar that it is the one running. */
export function runLabel(p: Procedure): string {
  return `Running “${p.label}”`;
}

function plainClick(e: MouseEvent<HTMLAnchorElement>): boolean {
  return !(
    e.defaultPrevented ||
    e.button !== 0 ||
    e.metaKey ||
    e.ctrlKey ||
    e.shiftKey ||
    e.altKey
  );
}

/** Go to the other level of the same model, remembering the choice. */
export function goToLevel(model: ModelDef, view: ModelView) {
  rememberView(view);
  navigate(modelPath(model.id, view));
}

export function ModelHeader({ model, view }: { model: ModelDef; view: ModelView }) {
  const family = FAMILIES.find((f) => f.id === model.family)!;
  const fid = FIDELITY[model.fidelity];
  return (
    <header className="model-head">
      <div className="model-head-text">
        <a
          className="back-link"
          href={href(`/#${family.id}`)}
          onClick={(e) => onLinkClick(e, `/#${family.id}`)}
        >
          <svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true">
            <path d="M7.5 2.5L4 6l3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          All models
        </a>
        <h1 className="model-title">{model.name}</h1>
        <p className="model-meta">
          <span className="model-family">{family.name}</span>
          <span className={`tag tag-${model.fidelity}`} title={fid.title}>
            {fid.label}
          </span>
          {model.checkpoints?.length ? (
            <span className="tag tag-trained" title="Weights trained offline in this engine">
              Trained weights included
            </span>
          ) : null}
        </p>
      </div>
      <nav className="level-switch" aria-label="Level of detail">
        {(['guide', 'lab'] as const).map((v) => (
          <a
            key={v}
            href={href(modelPath(model.id, v))}
            aria-current={view === v ? 'page' : undefined}
            className={view === v ? 'is-on' : ''}
            onClick={(e) => {
              if (!plainClick(e)) return;
              e.preventDefault();
              if (view !== v) goToLevel(model, v);
            }}
          >
            <span className="level-name">{v === 'guide' ? 'Guide' : 'Lab'}</span>
            <span className="level-desc">
              {v === 'guide' ? 'Watch it learn' : 'Every tensor and setting'}
            </span>
          </a>
        ))}
      </nav>
    </header>
  );
}

/** Input, training controls and the current numbers; sticks under the top bar. */
export function ActionBar({
  exp,
  model,
  onTrain,
  onStep,
  onReset,
  lead,
}: {
  exp: Experiment;
  model: ModelDef;
  onTrain?: (steps: number) => void;
  onStep?: () => void;
  onReset?: () => void;
  /** Placed before the input picker (the Lab's settings toggle). */
  lead?: ReactNode;
}) {
  const summary = useStore(exp.store, (s) => s.summary);
  const trace = useStore(exp.store, (s) => s.trace);
  const busy = useStore(exp.store, (s) => s.busy);
  const metric = trace ? Object.entries(trace.metrics)[0] : undefined;
  return (
    <div className="action-bar" data-busy={busy ? 'true' : 'false'}>
      <div className="action-bar-inner">
        {lead}
        <InputPicker exp={exp} model={model} />
        <TrainControls
          exp={exp}
          model={model}
          onTrain={onTrain}
          onStep={onStep}
          onReset={onReset}
        />
        <dl className="bar-readout" aria-live="polite">
          <div>
            <dt>
              <Term id="step">Step</Term>
            </dt>
            <dd className="num">{summary?.step ?? 0}</dd>
          </div>
          {trace?.loss !== undefined && (
            <div>
              <dt>
                <Term id="loss">Loss</Term>
              </dt>
              <dd className="num">{fmt(trace.loss, 4)}</dd>
            </div>
          )}
          {metric && (
            <div>
              <dt>
                <Term>{metricLabel(metric[0])}</Term>
              </dt>
              <dd className="num">{fmt(metric[1], 3)}</dd>
            </div>
          )}
        </dl>
        <span className="bar-status">{busy ? <Spinner label={busy} /> : null}</span>
      </div>
    </div>
  );
}

/** The most recent error, under the action bar. */
export function ErrorNote({ exp }: { exp: Experiment }) {
  const error = useStore(exp.store, (s) => s.error);
  if (!error) return null;
  return (
    <div className="ws-error" role="alert">
      <p>{error}</p>
    </div>
  );
}

/**
 * A procedure's arguments, its Run (or Stop) button, timing and scalar
 * results, and whether the result predates the latest change.
 */
export function ProcedureBar({
  exp,
  proc,
  args,
  onArgs,
}: {
  exp: Experiment;
  proc: Procedure;
  args?: Config;
  onArgs?: (args: Config) => void;
}) {
  const busy = useStore(exp.store, (s) => s.busy);
  const progress = useStore(exp.store, (s) => s.progress);
  const res = useStore(exp.store, (s) => s.procedures[proc.id]);
  const rev = useStore(exp.store, (s) => s.rev);
  const label = runLabel(proc);
  const running = busy === label;
  const stale = !!res && res.rev !== rev;
  const a = args ?? res?.args ?? proc.defaults ?? {};
  return (
    <div className="proc-bar">
      {proc.args?.length && onArgs ? (
        <ConfigFields
          inline
          fields={proc.args}
          config={a}
          disabled={!!busy}
          onChange={(k, v) => onArgs({ ...a, [k]: v })}
        />
      ) : null}
      <div className="proc-run">
        {running ? (
          <Button size="sm" onClick={() => exp.stop()}>
            Stop
          </Button>
        ) : (
          <Button
            size="sm"
            variant={!res || stale ? 'primary' : 'plain'}
            disabled={!!busy}
            onClick={() => exp.procedure(proc.id, a, label)}
          >
            {proc.label}
          </Button>
        )}
        {running && progress !== null && (
          <span className="muted num">{Math.round(progress * 100)}%</span>
        )}
        {res && !running && (
          <span className="muted num" title="Measured in this browser">
            {fmtMs(res.ms)}
          </span>
        )}
        {res?.result.scalars &&
          Object.entries(res.result.scalars).map(([k, v]) => (
            <span key={k} className="chip">
              {k} <span className="num">{fmt(v)}</span>
            </span>
          ))}
      </div>
      {res && (stale || res.stopped) && (
        <p className="proc-note">
          {res.stopped ? 'Stopped early: the result covers part of the run. ' : ''}
          {stale ? `Computed at step ${res.step}, before the latest change.` : ''}
        </p>
      )}
    </div>
  );
}

/** A Run button for an empty view whose data comes from a procedure. */
export function RunButton({
  exp,
  proc,
  args,
}: {
  exp: Experiment;
  proc: Procedure;
  args?: Config;
}) {
  const busy = useStore(exp.store, (s) => s.busy);
  return (
    <Button
      size="sm"
      variant="primary"
      disabled={!!busy}
      onClick={() => exp.procedure(proc.id, args ?? proc.defaults ?? {}, runLabel(proc))}
    >
      {proc.label}
    </Button>
  );
}
