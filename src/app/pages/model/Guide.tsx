import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Config, ModelDef, Procedure, ViewSpec } from '../../../models/types';
import type { Experiment } from '../../experiment/controller';
import { useStore } from '../../../state/store';
import { mathFor, useModuleGraph } from '../../experiment/Inspector';
import { lossSeries } from '../../experiment/Timeline';
import { OutputView, procedureOf } from '../../../viz/views';
import { LineChart } from '../../../viz/LineChart';
import { ErrorBoundary } from '../../../ui/ErrorBoundary';
import { Button, Card, EmptyState, Spinner } from '../../../ui/controls';
import { Tex } from '../../../ui/Tex';
import { describeModule } from '../../../viz/architecture';
import { fmtCount, fmtShape, shortCite } from '../../../ui/format';
import {
  ActionBar,
  ErrorNote,
  FIDELITY,
  ProcedureBar,
  RunButton,
  goToLevel,
  runLabel,
  visibleViews,
} from './shared';
import { NetworkCanvas } from './Lab';
import { Prose, Term } from '../../../ui/Term';

/** Procedures faster than this are re-run between training chunks so their views follow training. */
const FAST_MS = 300;

/** The views shown large: the guide's choice, or the first two that apply. */
function guideViews(model: ModelDef, config: Config | undefined): ViewSpec[] {
  const vis = visibleViews(model, config);
  const named = (model.guide?.views ?? [])
    .map((t) => vis.find((v) => v.title === t))
    .filter((v): v is ViewSpec => !!v);
  return named.length ? named : vis.slice(0, 2);
}

/**
 * Procedures the guide text names (“Label”) that feed none of the main views,
 * each with the first view that shows its result.
 */
function mentionedProcedures(
  model: ModelDef,
  config: Config | undefined,
  main: ViewSpec[],
): { proc: Procedure; view: ViewSpec }[] {
  const g = model.guide;
  if (!g) return [];
  const text = [g.what, g.watch, ...(g.try ?? []).map((t) => t.note)].join(' ');
  const shown = new Set(main.map(procedureOf));
  const vis = visibleViews(model, config);
  return (model.procedures ?? []).flatMap((proc) => {
    if (shown.has(proc.id) || !text.includes(`“${proc.label}”`)) return [];
    const view = vis.find((v) => procedureOf(v) === proc.id);
    return view ? [{ proc, view }] : [];
  });
}

/**
 * Keeps procedure results current. Those behind the main views run once when
 * missing; fast ones (main or mentioned, once run) run again after every
 * change. Slow ones are marked out of date by their toolbar instead.
 */
function useGuideProcedures(
  exp: Experiment,
  model: ModelDef,
  views: ViewSpec[],
  mentioned: Procedure[],
) {
  const key = views.map((v) => procedureOf(v) ?? '').join(',');
  const procs = useMemo(() => {
    const ids = [...new Set(key.split(',').filter(Boolean))];
    return ids
      .map((id) => model.procedures?.find((p) => p.id === id))
      .filter((p): p is Procedure => !!p);
  }, [key, model]);
  const mkey = mentioned.map((p) => p.id).join(',');
  const all = useMemo(
    () => [...procs, ...(model.procedures ?? []).filter((p) => mkey.split(',').includes(p.id))],
    [procs, mkey, model],
  );
  const status = useStore(exp.store, (s) => s.status);
  const busy = useStore(exp.store, (s) => s.busy);
  const live = useStore(exp.store, (s) => s.live);
  const rev = useStore(exp.store, (s) => s.rev);
  const results = useStore(exp.store, (s) => s.procedures);
  const tried = useRef(new Set<string>());

  useEffect(() => {
    if (status !== 'ready' || busy || live) return;
    const next = all.find((p) => {
      const r = results[p.id];
      if (!r && !procs.includes(p)) return false;
      if (r && (r.rev === rev || r.ms > FAST_MS)) return false;
      return !tried.current.has(`${p.id}@${rev}`);
    });
    if (!next) return;
    tried.current.add(`${next.id}@${rev}`);
    void exp.procedure(next.id, results[next.id]?.args ?? next.defaults ?? {}, runLabel(next));
  }, [exp, procs, all, status, busy, live, rev, results]);

  /** Re-run fast, out-of-date procedures between training chunks, while the effect above waits. */
  const refresh = useCallback(async () => {
    for (const p of all) {
      const r = exp.state.procedures[p.id];
      const k = `${p.id}@${exp.state.rev}`;
      if (r && r.rev !== exp.state.rev && r.ms <= FAST_MS && !tried.current.has(k)) {
        tried.current.add(k);
        await exp.procedure(p.id, r.args, runLabel(p));
      }
    }
  }, [exp, all]);
  return { procs: all, refresh };
}

export function GuidePage({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  const trace = useStore(exp.store, (s) => s.trace);
  const previous = useStore(exp.store, (s) => s.previous);
  const procedures = useStore(exp.store, (s) => s.procedures);
  const config = summary?.config;
  const views = useMemo(() => guideViews(model, config), [model, config]);
  const extra = useMemo(() => mentionedProcedures(model, config, views), [model, config, views]);
  const mentioned = useMemo(() => extra.map((e) => e.proc), [extra]);
  const { procs, refresh } = useGuideProcedures(exp, model, views, mentioned);
  const g = model.guide;
  const ctx = { trace, previous, procedures };
  const train = useCallback((n: number) => exp.trainLive(n, refresh), [exp, refresh]);
  const ref = model.references[0];

  return (
    <>
      <ActionBar exp={exp} model={model} onTrain={train} onReset={() => exp.open()} />
      <ErrorNote exp={exp} />
      <div className="guide">
        <section className="guide-intro" aria-labelledby="g-what">
          <div className="guide-intro-text">
            <h2 id="g-what" className="section-h">
              What it does
            </h2>
            <p className="lede">
              <Prose text={g?.what ?? model.summary} />
            </p>
            {ref && (
              <p className="guide-cite muted">
                From{' '}
                {ref.url ? (
                  <a href={ref.url} target="_blank" rel="noreferrer">
                    {ref.title}
                  </a>
                ) : (
                  ref.title
                )}
                , {shortCite(ref)}.
              </p>
            )}
          </div>
          <figure className="guide-eq">
            <Tex tex={model.signature} block />
            <figcaption>The defining equation</figcaption>
          </figure>
        </section>

        <section className="guide-section" aria-labelledby="g-watch">
          <h2 id="g-watch" className="section-h">
            Watch it learn
          </h2>
          {g?.watch && (
            <p className="guide-text">
              <Prose text={g.watch} />
            </p>
          )}
          <div className="card-grid">
            {views.map((v, i) => {
              const pid = procedureOf(v);
              const p = pid ? procs.find((x) => x.id === pid) : undefined;
              return (
                <ErrorBoundary key={`${i}:${v.title}`} label={`“${v.title}”`} resetKey={trace}>
                  <OutputView
                    spec={v}
                    ctx={ctx}
                    span={views.length === 1 ? 12 : 6}
                    toolbar={p ? <ProcedureBar exp={exp} proc={p} /> : undefined}
                    help={p?.description}
                    runner={p ? <RunButton exp={exp} proc={p} /> : undefined}
                  />
                </ErrorBoundary>
              );
            })}
            <LossCard exp={exp} model={model} onTrain={train} />
          </div>
          {extra.length > 0 && (
            <>
              <h3 className="sub-h">Experiments named above</h3>
              <p className="guide-text">
                Each of these runs a separate computation on the current weights when you press its
                button.
              </p>
              <div className="card-grid">
                {extra.map(({ proc, view }) => (
                  <ErrorBoundary key={proc.id} label={`“${view.title}”`} resetKey={trace}>
                    <OutputView
                      spec={view}
                      ctx={ctx}
                      span={6}
                      toolbar={<ProcedureBar exp={exp} proc={proc} />}
                      help={proc.description}
                      runner={<RunButton exp={exp} proc={proc} />}
                    />
                  </ErrorBoundary>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="guide-section" aria-labelledby="g-net">
          <h2 id="g-net" className="section-h">
            The network
          </h2>
          <p className="guide-text">
            Each box is one block of the computation, drawn from the operations the engine ran for
            the current input. Select a block to see what it computes.
          </p>
          <GuideNetwork exp={exp} model={model} />
        </section>

        {g?.try?.length ? <TryChanging exp={exp} model={model} refresh={refresh} /> : null}

        <section className="guide-section guide-further" aria-labelledby="g-further">
          <h2 id="g-further" className="section-h">
            Go further
          </h2>
          <div className="further-grid">
            <div className="further-lab">
              <h3 className="card-title">The lab</h3>
              <p>
                Every tensor and gradient of this computation, every setting, the training timeline
                step by step, and a check of the gradients against finite differences.
              </p>
              <Button variant="primary" onClick={() => goToLevel(model, 'lab')}>
                Open the lab
              </Button>
            </div>
            <div className="further-notes">
              <h3 className="card-title">What this version leaves out</h3>
              <p>
                <span className={`tag tag-${model.fidelity}`}>
                  <Term id="fidelity">{FIDELITY[model.fidelity].label}</Term>
                </span>{' '}
                {model.scaleNote ?? `${FIDELITY[model.fidelity].title}.`}
              </p>
              <ul>
                {model.limitations.map((l, i) => (
                  <li key={i}>
                    <Prose text={l} limit={2} />
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      </div>
    </>
  );
}

/** Training loss so far, and which weights the model started from. */
function LossCard({
  exp,
  model,
  onTrain,
}: {
  exp: Experiment;
  model: ModelDef;
  onTrain: (steps: number) => void;
}) {
  const history = useStore(exp.store, (s) => s.history);
  const summary = useStore(exp.store, (s) => s.summary);
  const provenance = useStore(exp.store, (s) => s.provenance);
  const busy = useStore(exp.store, (s) => s.busy);
  const live = useStore(exp.store, (s) => s.live);
  const series = useMemo(() => lossSeries(history), [history]);
  if (!summary || model.training.support === 'none') return null;
  const ck = model.checkpoints?.[0];
  const trained = !!summary.checkpoint;
  const status = trained
    ? `Trained weights${provenance ? `: ${provenance.steps.toLocaleString('en')} steps on ${provenance.dataset}, offline in this engine` : ''}${summary.step ? `, then ${summary.step} more here` : ''}.`
    : `Random initial weights (seed ${summary.seed})${summary.step ? `, trained ${summary.step} steps here` : ''}.`;
  const disabled = !!busy || !!live;
  return (
    <Card
      title="Loss"
      span={12}
      className="loss-card"
      help="The loss of each training batch: the number the optimiser lowers. Lower is better; its scale depends on the model."
      actions={
        trained ? (
          <Button size="sm" disabled={disabled} onClick={() => exp.reinit(summary.seed)}>
            Start from random weights
          </Button>
        ) : ck ? (
          <Button size="sm" disabled={disabled} onClick={() => exp.loadCheckpoint(ck.id)}>
            Load trained weights
          </Button>
        ) : undefined
      }
    >
      <p className="weights-status">{status}</p>
      {history.length ? (
        <LineChart
          series={series}
          xLabel="step"
          yLabel="loss"
          height={200}
          logY={history.every((r) => Object.values(r.losses).every((v) => v > 0))}
        />
      ) : (
        <EmptyState
          action={
            <Button
              variant="primary"
              disabled={disabled}
              onClick={() => onTrain(model.training.steps)}
            >
              Train {model.training.steps.toLocaleString('en')} steps
            </Button>
          }
        >
          No training on this page yet. Each step draws a batch of {summary.batchSize}, computes the
          loss, backpropagates and updates the weights; the views above redraw as it goes.
        </EmptyState>
      )}
    </Card>
  );
}

function GuideNetwork({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const trace = useStore(exp.store, (s) => s.trace);
  const selection = useStore(exp.store, (s) => s.selection);
  const busy = useStore(exp.store, (s) => s.busy);
  const graph = useModuleGraph(trace);
  const name =
    selection?.kind === 'module'
      ? selection.name
      : selection?.kind === 'node' && graph
        ? (graph.moduleOf.get(selection.id) ?? null)
        : null;
  const m = name && graph ? graph.byName.get(name) : undefined;
  const math = m ? mathFor(model, m.output.scope || m.name) : undefined;
  return (
    <Card className="guide-canvas">
      {graph ? (
        <NetworkCanvas exp={exp} model={model} graph={graph} mode="values" expandable={false} />
      ) : (
        <div className="canvas-placeholder">
          <Spinner label={busy ?? 'Loading'} />
        </div>
      )}
      <div className="guide-block" aria-live="polite">
        {m ? (
          <>
            <div className="guide-block-head">
              <h3 className="card-title">{m.label}</h3>
              <span className="muted">
                {describeModule(m)}; output {fmtShape(m.output.shape)}
                {m.paramCount ? `; ${fmtCount(m.paramCount)} parameters` : ''}
              </span>
            </div>
            {math && (
              <div className="math-block">
                <Tex tex={math.tex} block />
                {math.text && <p>{math.text}</p>}
              </div>
            )}
            <Button size="sm" onClick={() => goToLevel(model, 'lab')}>
              Inspect its values and gradients in the lab
            </Button>
          </>
        ) : (
          <p className="muted">Select a block to see its equation and size here.</p>
        )}
      </div>
    </Card>
  );
}

/** “Label: value” for each key of a configuration patch, in the words of the settings. */
function describePatch(model: ModelDef, patch: Config): string {
  return Object.entries(patch)
    .map(([k, v]) => {
      const f = model.configSchema.find((x) => x.key === k);
      if (!f) return `${k}: ${String(v)}`;
      const value =
        f.kind === 'select'
          ? (f.options.find((o) => o.value === v)?.label ?? String(v))
          : f.kind === 'bool'
            ? v
              ? 'on'
              : 'off'
            : String(v);
      return `${f.label}: ${value}`;
    })
    .join('; ');
}

function TryChanging({
  exp,
  model,
  refresh,
}: {
  exp: Experiment;
  model: ModelDef;
  refresh: () => Promise<void>;
}) {
  const busy = useStore(exp.store, (s) => s.busy);
  const live = useStore(exp.store, (s) => s.live);
  const config = useStore(exp.store, (s) => s.summary?.config);
  const [chosen, setChosen] = useState<number | null>(null);
  const tries = model.guide!.try!;
  const disabled = !!busy || !!live;
  const matches = (patch: Config) =>
    !!config && Object.entries(patch).every(([k, v]) => config[k] === v);
  // Forget the choice once the settings no longer show it (Reset, Lab changes).
  const applied = chosen !== null && matches(tries[chosen].patch) ? chosen : null;
  const isDefault = matches(model.defaults) && applied === null;
  return (
    <section className="guide-section" aria-labelledby="g-try">
      <div className="section-head">
        <h2 id="g-try" className="section-h">
          Try changing
        </h2>
        <Button
          variant="quiet"
          size="sm"
          disabled={disabled || isDefault}
          onClick={async () => {
            setChosen(null);
            await exp.open();
          }}
        >
          Restore defaults
        </Button>
      </div>
      <p className="guide-text">
        Each change starts again from the state this page opened in (default settings and the same
        starting weights), applies the change to the settings and, where stated, trains. Every
        result is computed here; the notes say what the same run gave when this text was written.
      </p>
      <div className="card-grid try-grid">
        {tries.map((t, i) => (
          <Card
            key={i}
            span={4}
            className={`try-card${applied === i ? ' is-applied' : ''}`}
            title={t.label}
          >
            <p>
              <Prose text={t.note} limit={2} />
            </p>
            <p className="try-patch mono">{describePatch(model, t.patch)}</p>
            <div className="try-actions">
              <Button
                variant={applied === i ? 'plain' : 'primary'}
                size="sm"
                disabled={disabled}
                onClick={() => {
                  setChosen(i);
                  void exp.replay(t.patch, t.train ?? 0, refresh);
                }}
              >
                {t.train ? `Apply and train ${t.train.toLocaleString('en')} steps` : 'Apply'}
              </Button>
              {applied === i && <span className="pill pill-ok">Applied</span>}
            </div>
          </Card>
        ))}
      </div>
    </section>
  );
}
