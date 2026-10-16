import { useEffect, useMemo, useState } from 'react';
import { FAMILIES, MODELS, findModel } from '../../models/catalog';
import type { Config, ModelDef } from '../../models/types';
import { Experiment } from '../experiment/controller';
import { useStore } from '../../state/store';
import { useModuleGraph } from '../experiment/Inspector';
import { ConfigFields } from '../experiment/Rail';
import { lossSeries } from '../experiment/Timeline';
import { LineChart, type LineSeries } from '../../viz/LineChart';
import { Histogram } from '../../viz/Histogram';
import { OutputView } from '../../viz/views';
import { moduleLabel } from '../../viz/architecture';
import {
  Button,
  Card,
  EmptyState,
  Field,
  NumberInput,
  Select,
  Spinner,
  Tabs,
} from '../../ui/controls';
import { fmt, fmtCount, fmtMs, metricLabel } from '../../ui/format';
import { navigate } from '../router';
import type { Trace } from '../../runtime/trace';

const SIDE = {
  a: { label: 'A', color: 'var(--ink)' },
  b: { label: 'B', color: 'var(--cmp)' },
} as const;
type Side = keyof typeof SIDE;
type CmpTab = 'measurements' | 'training' | 'blocks' | 'outputs';

function parseConfig(s: string | null): Config | undefined {
  if (!s) return undefined;
  try {
    const v = JSON.parse(s);
    return v && typeof v === 'object' ? (v as Config) : undefined;
  } catch {
    return undefined;
  }
}

/** Default pairing: the same network with two activations, so node-wise differences are defined. */
function defaults(params: URLSearchParams) {
  const a = findModel(params.get('a') ?? '') ?? findModel('mlp') ?? MODELS[0];
  const b = findModel(params.get('b') ?? '') ?? a;
  const ac = parseConfig(params.get('ac')) ?? {};
  const bc =
    parseConfig(params.get('bc')) ??
    (b.id === 'mlp' && !params.get('b') ? { activation: 'relu' } : {});
  const seed = Number(params.get('seed')) || 1;
  return { a, b, ac, bc, seed };
}

function totalFlops(t: Trace | null): number {
  return t ? t.nodes.reduce((s, n) => s + n.flops, 0) : 0;
}

function useSide(model: ModelDef, config: Config, seed: number) {
  // A new experiment whenever the model or the seed changes; configuration changes go through setConfig.
  const exp = useMemo(() => new Experiment(model.id), [model.id, seed]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    exp.open({ config, seed, checkpoint: null });
    return () => exp.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exp]);
  return exp;
}

const modelOptions = FAMILIES.flatMap((f) =>
  MODELS.filter((m) => m.family === f.id).map((m) => ({ value: m.id, label: m.name })),
);

function SidePanel({
  side,
  model,
  exp,
  onModel,
}: {
  side: Side;
  model: ModelDef;
  exp: Experiment;
  onModel: (id: string) => void;
}) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const error = useStore(exp.store, (s) => s.error);
  return (
    <section className={`cmp-side cmp-${side}`} aria-label={`Configuration ${SIDE[side].label}`}>
      <div className="cmp-side-head">
        <span className="cmp-badge" aria-hidden="true">
          {SIDE[side].label}
        </span>
        <Select
          value={model.id}
          options={modelOptions}
          onChange={onModel}
          ariaLabel={`Model ${SIDE[side].label}`}
        />
        {busy && <Spinner label={busy} />}
      </div>
      {summary && (
        <ConfigFields
          fields={model.configSchema}
          config={summary.config}
          disabled={!!busy}
          onChange={(k, v) => exp.setConfig({ [k]: v })}
        />
      )}
      {summary?.invalid && <p className="field-hint is-bad">{summary.invalid}</p>}
      {error && (
        <p className="field-hint is-bad" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

function MetricsTable({ a, b }: { a: Experiment; b: Experiment }) {
  const ta = useStore(a.store, (s) => s.trace);
  const tb = useStore(b.store, (s) => s.trace);
  const sa = useStore(a.store, (s) => s.summary);
  const sb = useStore(b.store, (s) => s.summary);
  const ha = useStore(a.store, (s) => s.history);
  const hb = useStore(b.store, (s) => s.history);
  const va = [...ha].reverse().find((r) => r.validation)?.validation;
  const vb = [...hb].reverse().find((r) => r.validation)?.validation;
  const metricNames = [
    ...new Set([...Object.keys(ta?.metrics ?? {}), ...Object.keys(tb?.metrics ?? {})]),
  ];
  const rows: {
    label: string;
    a?: number;
    b?: number;
    format: (v: number) => string;
  }[] = [
    {
      label: 'Parameters',
      a: sa?.paramCount,
      b: sb?.paramCount,
      format: fmtCount,
    },
    {
      label: 'FLOPs per forward pass',
      a: totalFlops(ta),
      b: totalFlops(tb),
      format: fmtCount,
    },
    {
      label: 'Forward time',
      a: ta?.timing.forwardMs,
      b: tb?.timing.forwardMs,
      format: fmtMs,
    },
    {
      label: 'Backward time',
      a: ta?.timing.backwardMs,
      b: tb?.timing.backwardMs,
      format: fmtMs,
    },
    { label: 'Steps trained', a: sa?.step, b: sb?.step, format: (v) => String(v) },
    {
      label: 'Loss',
      a: ta?.loss,
      b: tb?.loss,
      format: (v) => fmt(v, 4),
    },
    ...metricNames.map((m) => ({
      label: metricLabel(m),
      a: ta?.metrics[m],
      b: tb?.metrics[m],
      format: (v: number) => fmt(v, 4),
    })),
    {
      label: 'Validation loss',
      a: va?.loss,
      b: vb?.loss,
      format: (v) => fmt(v, 4),
    },
  ];
  return (
    <Card
      title="Measurements on the inspected example"
      help="Parameter and FLOP counts follow from the shapes. Times are measured in this tab, so they depend on this device and on what else it is doing."
      span={12}
    >
      <div className="table-wrap">
        <table className="data-table cmp-table">
          <thead>
            <tr>
              <th scope="col">Quantity</th>
              <th scope="col" className="r cmp-a-text">
                A
              </th>
              <th scope="col" className="r cmp-b-text">
                B
              </th>
              <th scope="col" className="r">
                B / A
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <th scope="row">{r.label}</th>
                <td className="r num">{r.a !== undefined ? r.format(r.a) : '—'}</td>
                <td className="r num">{r.b !== undefined ? r.format(r.b) : '—'}</td>
                <td className="r num muted">
                  {r.a && r.b !== undefined && Number.isFinite(r.b / r.a)
                    ? `${fmt(r.b / r.a, 3)}×`
                    : ''}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Trajectories({ a, b }: { a: Experiment; b: Experiment }) {
  const ha = useStore(a.store, (s) => s.history);
  const hb = useStore(b.store, (s) => s.history);
  if (!ha.length && !hb.length)
    return (
      <Card title="No training yet" span={12}>
        <EmptyState>
          Press Train both: A and B train from the same seed on the same batches, and their loss
          curves are drawn here on one axis.
        </EmptyState>
      </Card>
    );
  const loss: LineSeries[] = [
    ...lossSeries(ha, 'A ', ['var(--ink)']),
    ...lossSeries(hb, 'B ', ['var(--cmp)']).map((s) => ({ ...s, dashed: !s.markers })),
  ];
  const grad: LineSeries[] = [
    {
      id: 'ga',
      label: 'A ‖∇θ L‖',
      color: 'var(--ink)',
      points: ha.map((r) => ({ x: r.step, y: r.gradNorm })),
    },
    {
      id: 'gb',
      label: 'B ‖∇θ L‖',
      color: 'var(--cmp)',
      dashed: true,
      points: hb.map((r) => ({ x: r.step, y: r.gradNorm })),
    },
  ];
  return (
    <div className="card-grid">
      <Card title="Loss" span={6} help="A solid, B dashed; markers are held-out losses.">
        <LineChart series={loss} logY xLabel="step" yLabel="loss" height={220} />
      </Card>
      <Card
        title="Gradient norm"
        span={6}
        help="‖∇θ L‖ over all trainable parameters at each step, before clipping."
      >
        <LineChart series={grad} logY xLabel="step" yLabel="‖∇θ L‖" height={220} />
      </Card>
    </div>
  );
}

/** Module-by-module comparison. Defined only when both traces have the same modules and output shapes. */
function NodeDiffs({ a, b }: { a: Experiment; b: Experiment }) {
  const ta = useStore(a.store, (s) => s.trace);
  const tb = useStore(b.store, (s) => s.trace);
  const ga = useModuleGraph(ta);
  const gb = useModuleGraph(tb);
  const [sel, setSel] = useState<string | null>(null);
  if (!ga || !gb) return null;
  const pairs = ga.modules
    .map((m) => ({ m, n: gb.byName.get(m.name) }))
    .filter(({ m, n }) => n && n.output.value.length === m.output.value.length);
  if (!pairs.length)
    return (
      <Card title="No blocks in common" span={12}>
        <EmptyState>
          A and B have no block with the same name and output size. Pick two settings of the same
          model to compare them block by block.
        </EmptyState>
      </Card>
    );
  const rows = pairs.map(({ m, n }) => {
    let maxAbs = 0;
    let ref = 0;
    for (let i = 0; i < m.output.value.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(m.output.value[i] - n!.output.value[i]));
      ref = Math.max(ref, Math.abs(m.output.value[i]));
    }
    return { name: m.name, a: m, b: n!, maxAbs, rel: ref ? maxAbs / ref : 0 };
  });
  const chosen = rows.find((r) => r.name === sel) ?? rows[0];
  const maxRel = Math.max(...rows.map((r) => r.rel), 1e-12);
  return (
    <div className="card-grid">
      <Card
        title="Block outputs, A against B"
        span={6}
        help="Largest absolute difference between the two outputs of each block, and that difference relative to the largest value of A. Select a block to compare the distributions."
      >
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">Block output</th>
                <th scope="col" className="r">
                  max |A − B|
                </th>
                <th scope="col">relative to max |A|</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className={r.name === chosen.name ? 'is-selected' : ''}>
                  <th scope="row">
                    <button type="button" className="linkish" onClick={() => setSel(r.name)}>
                      {moduleLabel(r.name)}
                    </button>
                  </th>
                  <td className="r num">{fmt(r.maxAbs, 3)}</td>
                  <td>
                    <span
                      className="diff-bar"
                      style={{ width: `${(r.rel / maxRel) * 100}%` }}
                      aria-label={fmt(r.rel, 3)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card
        title={`${moduleLabel(chosen.name)}: distributions`}
        span={6}
        help={`Histograms of the ${chosen.a.output.value.length} output values of each side, and of the gradients of the loss with respect to them.`}
      >
        <figure className="cmp-hist">
          <figcaption>Output values</figcaption>
          <Histogram
            width={420}
            height={130}
            series={[
              { data: chosen.a.output.value, color: 'var(--ink)', label: 'A' },
              { data: chosen.b.output.value, color: 'var(--cmp)', label: 'B' },
            ]}
          />
          {chosen.a.output.grad && chosen.b.output.grad && (
            <>
              <figcaption>Gradients ∂L/∂output</figcaption>
              <Histogram
                width={420}
                height={110}
                symmetric
                series={[
                  { data: chosen.a.output.grad, color: 'var(--ink)', label: 'A' },
                  { data: chosen.b.output.grad, color: 'var(--cmp)', label: 'B' },
                ]}
              />
            </>
          )}
        </figure>
      </Card>
    </div>
  );
}

function Outputs({ exp, model, side }: { exp: Experiment; model: ModelDef; side: Side }) {
  const trace = useStore(exp.store, (s) => s.trace);
  const previous = useStore(exp.store, (s) => s.previous);
  const procedures = useStore(exp.store, (s) => s.procedures);
  const summary = useStore(exp.store, (s) => s.summary);
  return (
    <div className={`card-grid cmp-outputs cmp-${side}`}>
      {model.views
        .filter((v) => !v.when || (summary && v.when(summary.config)))
        .map((v, i) => (
          <OutputView key={i} spec={v} ctx={{ trace, previous, procedures }} span={12} />
        ))}
    </div>
  );
}

export function ComparePage({ params }: { params: URLSearchParams }) {
  const init = useMemo(() => defaults(params), []); // eslint-disable-line react-hooks/exhaustive-deps
  const [ids, setIds] = useState({ a: init.a.id, b: init.b.id });
  const [seed, setSeed] = useState(init.seed);
  const [steps, setSteps] = useState(200);
  const ma = findModel(ids.a)!;
  const mb = findModel(ids.b)!;
  const a = useSide(ma, ids.a === init.a.id ? init.ac : {}, seed);
  const b = useSide(mb, ids.b === init.b.id ? init.bc : {}, seed);
  const sa = useStore(a.store, (s) => s.summary);
  const sb = useStore(b.store, (s) => s.summary);
  const busyA = useStore(a.store, (s) => s.busy);
  const busyB = useStore(b.store, (s) => s.busy);
  const [training, setTraining] = useState(false);
  const [tab, setTab] = useState<CmpTab>('measurements');

  // Keep the address shareable.
  useEffect(() => {
    if (!sa || !sb) return;
    const q = new URLSearchParams({
      a: ids.a,
      b: ids.b,
      ac: JSON.stringify(sa.config),
      bc: JSON.stringify(sb.config),
      seed: String(seed),
    });
    navigate(`/compare?${q}`, { replace: true });
  }, [sa, sb, ids, seed]);

  const both = async (fn: (e: Experiment) => Promise<void>) => {
    await Promise.all([fn(a), fn(b)]);
  };
  const runProcs = () =>
    both(async (e) => {
      const m = e === a ? ma : mb;
      for (const p of m.procedures ?? [])
        if (!p.args?.length) await e.procedure(p.id, p.defaults ?? {});
    });

  const ha = useStore(a.store, (s) => s.history);
  const hb = useStore(b.store, (s) => s.history);
  const canTrain = ma.training.support !== 'none' && mb.training.support !== 'none';
  const tabs: { id: CmpTab; label: string; count?: number }[] = [
    { id: 'measurements', label: 'Measurements' },
    { id: 'training', label: 'Training', count: Math.max(ha.length, hb.length) || undefined },
    { id: 'blocks', label: 'Block by block' },
    { id: 'outputs', label: 'Outputs' },
  ];

  return (
    <main className="page compare" id="main">
      <header className="page-head">
        <h1 className="page-h">Compare</h1>
        <p className="page-lede">
          Two models, or two settings of one model, opened with random weights from the same seed
          and trained on the same batches.
        </p>
      </header>

      <div className="cmp-sides">
        <SidePanel side="a" model={ma} exp={a} onModel={(id) => setIds({ ...ids, a: id })} />
        <SidePanel side="b" model={mb} exp={b} onModel={(id) => setIds({ ...ids, b: id })} />
      </div>

      <div className="cmp-controls">
        {training ? (
          <Button
            onClick={() => {
              a.stop();
              b.stop();
            }}
          >
            Pause
          </Button>
        ) : (
          <Button
            variant="primary"
            disabled={!sa || !sb || !!busyA || !!busyB || !canTrain}
            onClick={async () => {
              setTraining(true);
              setTab('training');
              await both((e) => e.train(steps));
              setTraining(false);
              await runProcs();
            }}
          >
            Train both
          </Button>
        )}
        <NumberInput
          value={steps}
          onCommit={(v) => setSteps(Math.max(1, Math.round(v)))}
          min={1}
          max={5000}
          step={50}
          integer
          width={64}
          ariaLabel="Steps"
        />
        <span className="bar-label">steps</span>
        <Button
          disabled={!sa || !sb || !!busyA || !!busyB || !canTrain}
          onClick={() => both((e) => e.step())}
        >
          Step
        </Button>
        {ma.procedures?.length || mb.procedures?.length ? (
          <Button variant="quiet" disabled={!!busyA || !!busyB} onClick={runProcs}>
            Run experiments
          </Button>
        ) : null}
        <span className="cmp-seed">
          <Field label="Seed" inline>
            <NumberInput
              value={seed}
              onCommit={(v) => setSeed(Math.max(0, Math.round(v)))}
              min={0}
              step={1}
              integer
              width={64}
              ariaLabel="Seed"
            />
          </Field>
        </span>
      </div>

      <Tabs tabs={tabs} value={tab} onChange={setTab} label="Comparison views" />
      <div className="cmp-panel" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === 'measurements' && <MetricsTable a={a} b={b} />}
        {tab === 'training' && <Trajectories a={a} b={b} />}
        {tab === 'blocks' && <NodeDiffs a={a} b={b} />}
        {tab === 'outputs' && (
          <div className="cmp-sides">
            <Outputs exp={a} model={ma} side="a" />
            <Outputs exp={b} model={mb} side="b" />
          </div>
        )}
      </div>
    </main>
  );
}
