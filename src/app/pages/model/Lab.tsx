import { useMemo, useState } from 'react';
import type { Config, ModelDef, Procedure } from '../../../models/types';
import type { Experiment } from '../../experiment/controller';
import { useStore } from '../../../state/store';
import {
  NetworkGraph,
  NetworkLegend,
  type GraphMode,
  type ModuleDiff,
} from '../../../viz/NetworkGraph';
import {
  GradientCheckCard,
  Inspector,
  NetworkSummary,
  mathFor,
  useModuleGraph,
} from '../../experiment/Inspector';
import {
  ComputationPanel,
  ConfigPanel,
  DataPanel,
  OptimizerPanel,
  WeightsPanel,
} from '../../experiment/Rail';
import { SavedPanel } from '../../experiment/SavedPanel';
import { Timeline } from '../../experiment/Timeline';
import { OutputView, procedureOf } from '../../../viz/views';
import { ErrorBoundary } from '../../../ui/ErrorBoundary';
import { Button, Card, EmptyState, Section, Segmented, Spinner, Tabs } from '../../../ui/controls';
import { Tex } from '../../../ui/Tex';
import { Prose, Term } from '../../../ui/Term';
import { type Module, type ModuleGraph } from '../../../viz/architecture';
import { fmt } from '../../../ui/format';
import { ActionBar, ErrorNote, FIDELITY, ProcedureBar, visibleViews } from './shared';

type Tab = 'network' | 'outputs' | 'training' | 'check' | 'notes';

function wideScreen(): boolean {
  try {
    return window.matchMedia('(min-width: 1100px)').matches;
  } catch {
    return true;
  }
}

export function LabPage({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const [tab, setTab] = useState<Tab>('network');
  const [drawer, setDrawer] = useState(wideScreen);
  const summary = useStore(exp.store, (s) => s.summary);
  const history = useStore(exp.store, (s) => s.history);
  const gradcheck = useStore(exp.store, (s) => s.gradcheck);
  const viewCount = visibleViews(model, summary?.config).length;
  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: 'network', label: 'Network' },
    { id: 'outputs', label: 'Outputs', count: viewCount },
    ...(model.training.support !== 'none'
      ? [{ id: 'training' as const, label: 'Training', count: history.length || undefined }]
      : []),
    {
      id: 'check',
      label: gradcheck ? `Gradient check ${gradcheck.pass ? '✓' : '✕'}` : 'Gradient check',
    },
    { id: 'notes', label: 'Notes' },
  ];
  return (
    <>
      <ActionBar
        exp={exp}
        model={model}
        lead={
          <Button
            variant="quiet"
            className={`drawer-toggle${drawer ? ' is-on' : ''}`}
            aria-expanded={drawer}
            aria-controls="lab-settings"
            onClick={() => setDrawer(!drawer)}
          >
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <path
                d="M2 4h12M2 8h12M2 12h12"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <circle cx="5" cy="4" r="1.6" fill="var(--surface)" stroke="currentColor" />
              <circle cx="11" cy="8" r="1.6" fill="var(--surface)" stroke="currentColor" />
              <circle cx="7" cy="12" r="1.6" fill="var(--surface)" stroke="currentColor" />
            </svg>
            Settings
          </Button>
        }
      />
      <ErrorNote exp={exp} />
      <div className={`lab${drawer ? ' has-drawer' : ''}`}>
        {drawer && (
          <aside id="lab-settings" className="lab-drawer" aria-label="Settings">
            <ConfigPanel exp={exp} model={model} />
            <ComputationPanel exp={exp} model={model} />
            <DataPanel exp={exp} model={model} />
            <OptimizerPanel exp={exp} model={model} />
            <WeightsPanel exp={exp} model={model} />
            <SavedPanel exp={exp} model={model} />
          </aside>
        )}
        <div className="lab-main">
          <Tabs tabs={tabs} value={tab} onChange={setTab} label="Parts of the lab" />
          <div
            className={`lab-panel lab-panel-${tab}`}
            role="tabpanel"
            id={`panel-${tab}`}
            aria-labelledby={`tab-${tab}`}
          >
            {tab === 'network' && <NetworkTab exp={exp} model={model} />}
            {tab === 'outputs' && <OutputsTab exp={exp} model={model} />}
            {tab === 'training' && <Timeline exp={exp} />}
            {tab === 'check' && (
              <div className="card-grid">
                <GradientCheckCard exp={exp} model={model} />
              </div>
            )}
            {tab === 'notes' && <NotesTab exp={exp} model={model} />}
          </div>
        </div>
      </div>
    </>
  );
}

/** Largest change of each block output since the previous computation. */
export function useModuleDiffs(graph: ModuleGraph | null, exp: Experiment) {
  const previous = useStore(exp.store, (s) => s.previous);
  return useMemo(() => {
    if (!graph || !previous) return null;
    const prev = new Map(previous.nodes.map((n) => [n.key, n]));
    const out = new Map<string, ModuleDiff>();
    for (const m of graph.modules) {
      const p = prev.get(m.output.key);
      if (!p || p.value.length !== m.output.value.length) continue;
      let maxAbs = 0;
      let ref = 0;
      for (let i = 0; i < p.value.length; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(m.output.value[i] - p.value[i]));
        ref = Math.max(ref, Math.abs(p.value[i]));
      }
      out.set(m.name, { maxAbs, rel: ref > 0 ? maxAbs / ref : 0 });
    }
    return out;
  }, [graph, previous]);
}

/** First sentence of the block’s explanation in the model’s equations. */
function blockNoteFor(model: ModelDef) {
  return (m: Module) => {
    const text = mathFor(model, m.output.scope || m.name)?.text;
    if (!text) return undefined;
    const end = text.search(/\.(\s|$)/);
    return end > 0 && end < 220 ? text.slice(0, end + 1) : text.length <= 220 ? text : undefined;
  };
}

/** The computation graph, with selection and per-block changes. */
export function NetworkCanvas({
  exp,
  model,
  graph,
  mode,
  expandable = true,
}: {
  exp: Experiment;
  model: ModelDef;
  graph: ModuleGraph;
  mode: GraphMode;
  expandable?: boolean;
}) {
  const selection = useStore(exp.store, (s) => s.selection);
  const diffs = useModuleDiffs(graph, exp);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const note = useMemo(() => blockNoteFor(model), [model]);
  const selectedModule =
    selection?.kind === 'module'
      ? selection.name
      : selection?.kind === 'node'
        ? (graph.moduleOf.get(selection.id) ?? null)
        : null;
  return (
    <NetworkGraph
      graph={graph}
      selectedModule={selectedModule}
      selectedNode={selection?.kind === 'node' ? selection.id : null}
      onSelectModule={(name) => exp.select({ kind: 'module', name })}
      onSelectNode={(id) => exp.select({ kind: 'node', id })}
      expanded={expandable ? expanded : new Set()}
      onToggleExpand={(name) => {
        if (!expandable) return;
        const next = new Set(expanded);
        if (next.has(name)) next.delete(name);
        else next.add(name);
        setExpanded(next);
      }}
      mode={mode}
      diffs={diffs}
      blockNote={note}
    />
  );
}

function NetworkTab({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const trace = useStore(exp.store, (s) => s.trace);
  const selection = useStore(exp.store, (s) => s.selection);
  const busy = useStore(exp.store, (s) => s.busy);
  const graph = useModuleGraph(trace);
  const [mode, setMode] = useState<GraphMode>('values');
  const feedback = !!graph?.edges.some((e) => e.feedback);
  const stopGrad = !!graph?.edges.some((e) => e.stopGrad);
  const residual = !!graph?.modules.some((m) => m.residual);
  return (
    <div className="lab-network">
      <Card
        className="network-card"
        title={<Term id="graph">Computation graph</Term>}
        help="Each block groups the operations recorded in the last forward pass under one name, in execution order, from the input at the top to the loss at the bottom. The tile in a block is its output tensor. Hover over a block for what it does; select it for its equation, parameters and tensors. Gradients shows ∂L/∂y instead, with arrows pointing the way the backward pass runs."
        actions={
          <Segmented
            size="sm"
            ariaLabel="Show"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'values', label: 'Values' },
              {
                value: 'gradients',
                label: 'Gradients',
                title: 'Tiles and lines show ∂L/∂y, the gradient of the loss at each block output',
              },
            ]}
          />
        }
      >
        {graph ? (
          <>
            <NetworkCanvas exp={exp} model={model} graph={graph} mode={mode} />
            <NetworkLegend
              mode={mode}
              feedback={feedback}
              stopGrad={stopGrad}
              residual={residual}
            />
          </>
        ) : (
          <div className="canvas-placeholder">
            <Spinner label={busy ?? 'Loading'} />
          </div>
        )}
      </Card>
      <aside className="net-details" aria-label={selection ? 'Selected block' : 'Overview'}>
        {selection ? (
          <Inspector exp={exp} model={model} graph={graph} />
        ) : (
          trace && <Overview exp={exp} graph={graph} />
        )}
      </aside>
    </div>
  );
}

/** Numbers of the current computation and the blocks that changed most. */
function Overview({ exp, graph }: { exp: Experiment; graph: ModuleGraph | null }) {
  const trace = useStore(exp.store, (s) => s.trace)!;
  const previous = useStore(exp.store, (s) => s.previous);
  const cause = useStore(exp.store, (s) => s.cause);
  const diffs = useModuleDiffs(graph, exp);
  const changed = [...(diffs ?? [])]
    .filter(([, d]) => d.maxAbs > 0)
    .sort((a, b) => b[1].maxAbs - a[1].maxAbs);
  return (
    <div className="inspector overview">
      <header className="insp-head">
        <p className="insp-kicker">Last forward pass</p>
        <h2>Overview</h2>
        {cause && <p className="muted">{cause}</p>}
      </header>
      <NetworkSummary exp={exp} trace={trace} className="ov-stats" />
      <div className="insp-sub">
        <h3>
          <Term id="changed">What changed</Term>
        </h3>
        {!previous ? (
          <p className="muted">
            Step, train or change a setting; the blocks it affects appear here.
          </p>
        ) : changed.length === 0 ? (
          <p className="muted">No block output changed.</p>
        ) : (
          <ol className="op-list">
            {changed.slice(0, 8).map(([name, d]) => (
              <li key={name}>
                <button type="button" onClick={() => exp.select({ kind: 'module', name })}>
                  <span>{graph?.byName.get(name)?.label ?? name}</span>
                  <span className="num">max |Δ| {fmt(d.maxAbs, 3)}</span>
                </button>
              </li>
            ))}
          </ol>
        )}
      </div>
      <p className="ov-hint muted">
        Select a block for its equation, operations, parameters and tensors.
      </p>
    </div>
  );
}

/** Views grouped by where their data comes from: the forward pass, or one procedure each. */
function OutputsTab({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const trace = useStore(exp.store, (s) => s.trace);
  const previous = useStore(exp.store, (s) => s.previous);
  const procedures = useStore(exp.store, (s) => s.procedures);
  const summary = useStore(exp.store, (s) => s.summary);
  const [args, setArgs] = useState<Record<string, Config>>(() =>
    Object.fromEntries((model.procedures ?? []).map((p) => [p.id, { ...(p.defaults ?? {}) }])),
  );
  const ctx = { trace, previous, procedures };
  const views = visibleViews(model, summary?.config);
  const forward = views.filter((v) => !procedureOf(v));
  const byProc = (p: Procedure) => views.filter((v) => procedureOf(v) === p.id);
  const card = (v: (typeof views)[number], i: number, runner?: string) => (
    <ErrorBoundary key={`${i}:${v.title}`} label={`“${v.title}”`} resetKey={trace}>
      <OutputView
        spec={v}
        ctx={ctx}
        runner={runner ? <span className="muted">Run “{runner}” above.</span> : undefined}
      />
    </ErrorBoundary>
  );
  return (
    <div className="outputs">
      {forward.length > 0 && (
        <Section
          title="From the forward pass"
          desc="Recomputed for the current input after every step and every change."
        >
          {forward.map((v, i) => card(v, i))}
        </Section>
      )}
      {(model.procedures ?? []).map((p) => {
        const vs = byProc(p);
        const r = procedures[p.id];
        return (
          <Section
            key={p.id}
            id={`proc-${p.id}`}
            title={p.label}
            desc={<Prose text={p.description} />}
          >
            <div className="proc-head" style={{ gridColumn: '1 / -1' }}>
              <ProcedureBar
                exp={exp}
                proc={p}
                args={args[p.id]}
                onArgs={(a) => setArgs({ ...args, [p.id]: a })}
              />
            </div>
            {vs.length > 0
              ? vs.map((v, i) => card(v, i, p.label))
              : r?.result.text && (
                  <Card title="Result" span={12}>
                    <pre className="generated">{r.result.text}</pre>
                  </Card>
                )}
            {vs.length === 0 && r && !r.result.text && !r.result.scalars && (
              <Card title="Result" span={12}>
                <EmptyState>
                  Computed {Object.keys(r.result.tensors).length} tensor(s); no view on this page
                  shows them for the current settings.
                </EmptyState>
              </Card>
            )}
          </Section>
        );
      })}
    </div>
  );
}

function NotesTab({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  const provenance = useStore(exp.store, (s) => s.provenance);
  const ds = summary
    ? (model.datasets.find((d) => d.id === summary.datasetId) ?? model.datasets[0])
    : model.datasets[0];
  const fid = FIDELITY[model.fidelity];
  return (
    <div className="notes">
      <section>
        <h2 className="notes-h">Defining equation</h2>
        <div className="notes-eq">
          <Tex tex={model.signature} block />
        </div>
        {model.guide?.what && (
          <p>
            <Prose text={model.guide.what} />
          </p>
        )}
      </section>
      <section>
        <h2 className="notes-h">Scale</h2>
        <p>
          <span className={`tag tag-${model.fidelity}`}>
            <Term id="fidelity">{fid.label}</Term>
          </span>{' '}
          {fid.title}.
        </p>
        {model.scaleNote && <p>{model.scaleNote}</p>}
      </section>
      <section>
        <h2 className="notes-h">Limitations</h2>
        <ul>
          {model.limitations.map((l, i) => (
            <li key={i}>
              <Prose text={l} limit={2} />
            </li>
          ))}
        </ul>
      </section>
      {model.training.note && (
        <section>
          <h2 className="notes-h">Training</h2>
          <p>{model.training.note}</p>
        </section>
      )}
      {ds && (
        <section>
          <h2 className="notes-h">Data</h2>
          <p>
            {ds.name}. {ds.description}
          </p>
          {ds.license && <p className="muted">{ds.license}</p>}
        </section>
      )}
      {provenance && (
        <section>
          <h2 className="notes-h">Trained weights</h2>
          <p>
            Trained in this engine by <code>{provenance.script}</code> on {provenance.dataset}:{' '}
            {provenance.steps.toLocaleString('en')} steps, seed {provenance.seed}.
          </p>
          <dl className="notes-metrics">
            {Object.entries(provenance.metrics).map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd className="num">{fmt(v, 4)}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      <section>
        <h2 className="notes-h">References</h2>
        <ul className="refs">
          {model.references.map((r, i) => (
            <li key={i}>
              {r.authors} ({r.year}).{' '}
              {r.url ? (
                <a href={r.url} target="_blank" rel="noreferrer">
                  {r.title}
                </a>
              ) : (
                r.title
              )}
              .
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
