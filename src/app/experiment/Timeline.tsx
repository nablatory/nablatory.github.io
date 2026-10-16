import { useMemo, useState } from 'react';
import { useStore } from '../../state/store';
import type { Experiment } from './controller';
import { LineChart, type LineSeries } from '../../viz/LineChart';
import { TensorView } from '../../viz/TensorView';
import { fmt, fmtMs, metricLabel } from '../../ui/format';
import { Button, Card, EmptyState, Section } from '../../ui/controls';
import { Term } from '../../ui/Term';
import type { TrainRecord } from '../../runtime/session';

const PHASE_COLORS = ['var(--ink)', 'var(--cmp)', 'var(--neg)', 'var(--pos)'];

export function lossSeries(
  history: TrainRecord[],
  prefix = '',
  colors = PHASE_COLORS,
): LineSeries[] {
  const phases = [...new Set(history.flatMap((r) => Object.keys(r.losses)))];
  // One objective: the noisy per-batch line in grey under the held-out markers in ink.
  const single = phases.length === 1 && colors === PHASE_COLORS;
  const out: LineSeries[] = phases.map((p, i) => ({
    id: `${prefix}${p}`,
    label: `${prefix}${phases.length === 1 ? 'training loss' : `${p} loss`}`,
    color: single ? 'var(--ink-3)' : colors[i % colors.length],
    points: history.map((r) => ({ x: r.step, y: r.losses[p] })).filter((q) => Number.isFinite(q.y)),
  }));
  const val = history.filter((r) => r.validation);
  if (val.length) {
    out.push({
      id: `${prefix}val`,
      label: `${prefix}validation loss`,
      color: colors[0],
      markers: true,
      points: val.map((r) => ({ x: r.step, y: r.validation!.loss })),
    });
  }
  return out;
}

export function metricSeries(history: TrainRecord[]): LineSeries[] {
  const metricNames = [...new Set(history.flatMap((r) => Object.keys(r.metrics)))];
  return metricNames.flatMap((m, i) => {
    const s: LineSeries[] = [
      {
        id: m,
        label: `${metricLabel(m)}, training batch`,
        color: PHASE_COLORS[(i + 1) % 4],
        points: history
          .map((r) => ({ x: r.step, y: r.metrics[m] }))
          .filter((q) => Number.isFinite(q.y)),
      },
    ];
    const v = history.filter((r) => r.validation?.metrics[m] !== undefined);
    if (v.length)
      s.push({
        id: `v${m}`,
        label: `${metricLabel(m)}, validation`,
        color: PHASE_COLORS[(i + 1) % 4],
        markers: true,
        points: v.map((r) => ({ x: r.step, y: r.validation!.metrics[m] })),
      });
    return s;
  });
}

export function Timeline({ exp, onStep }: { exp: Experiment; onStep?: () => void }) {
  const history = useStore(exp.store, (s) => s.history);
  const detail = useStore(exp.store, (s) => s.stepDetail);
  const busy = useStore(exp.store, (s) => s.busy);
  const latest = history[history.length - 1];
  const metrics = useMemo(() => metricSeries(history), [history]);

  if (!history.length) {
    return (
      <section className="timeline" aria-label="Training">
        <Card title="No training yet">
          <EmptyState
            action={
              <Button
                variant="primary"
                disabled={!!busy}
                onClick={() => (onStep ? onStep() : exp.step())}
              >
                Run one step
              </Button>
            }
          >
            One step runs the forward pass, the loss, the backward pass and the parameter update,
            and records each of them here.
          </EmptyState>
        </Card>
      </section>
    );
  }

  return (
    <div className="timeline" aria-label="Training timeline">
      <Section
        title="Loss and metrics"
        desc="One point per training batch; markers are values on held-out data."
      >
        <Card
          title={<Term id="loss">Loss</Term>}
          span={metrics.length ? 7 : 12}
          help="Loss of each training batch; markers show the held-out loss where the model evaluates one."
          actions={
            latest && (
              <span className="muted" title="Measured in this browser">
                {fmtMs(latest.ms)} per step
              </span>
            )
          }
        >
          <LineChart
            series={lossSeries(history)}
            xLabel="step"
            yLabel="loss"
            logY={history.every((r) => Object.values(r.losses).every((v) => v > 0))}
            height={220}
          />
        </Card>
        {metrics.length > 0 && (
          <Card
            title="Metrics"
            span={5}
            help="What the model reports besides the loss, on each training batch; markers show the held-out value where the model evaluates one."
          >
            <LineChart xLabel="step" height={220} series={metrics} />
          </Card>
        )}
      </Section>
      {latest && (
        <Section
          title={<Term id="gradientFlow">Gradients</Term>}
          desc="How large the gradients are, overall and for each group of parameters."
        >
          <GradientFlow history={history} />
        </Section>
      )}
      {detail && (
        <Section
          title="Last step in detail"
          desc="Every stage of the most recent step: forward pass, loss, backward pass, update."
        >
          <StepDetailView exp={exp} />
        </Section>
      )}
    </div>
  );
}

function GradientFlow({ history }: { history: TrainRecord[] }) {
  const latest = history[history.length - 1];
  const names = Object.keys(latest.params);
  const maxG = Math.max(...names.map((n) => latest.params[n].gradNorm), 1e-12);
  const logBar = (v: number, m: number) => (v > 0 ? Math.max(0.02, 1 + Math.log10(v / m) / 6) : 0);
  return (
    <>
      <Card
        title="Gradient norm"
        span={5}
        help="Global norm of the gradient over all trainable parameters, before clipping."
      >
        <LineChart
          height={200}
          xLabel="step"
          yLabel="‖∇θ‖"
          logY
          series={[
            {
              id: 'g',
              label: 'global gradient norm',
              color: 'var(--grad)',
              points: history.map((r) => ({ x: r.step, y: r.gradNorm })),
            },
          ]}
        />
      </Card>
      <Card
        title={`Gradient flow at step ${latest.step}`}
        span={7}
        help="Very small update ratios (≲10⁻⁴) indicate a layer that is barely learning; large ones (≳10⁻¹) indicate instability."
      >
        <div className="gradflow">
          <p className="gradflow-head">
            <span>parameter</span>
            <span>‖∂L/∂θ‖ (log scale, 6 decades)</span>
            <span>‖Δθ‖ / ‖θ‖</span>
          </p>
          {names.map((n) => {
            const p = latest.params[n];
            const ratio = p.weightNorm > 0 ? p.updateNorm / p.weightNorm : 0;
            return (
              <div key={n} className="gradflow-row">
                <span className="gradflow-name" title={n}>
                  {n}
                </span>
                <span className="gradflow-track">
                  <span
                    className="gradflow-bar"
                    style={{ width: `${logBar(p.gradNorm, maxG) * 100}%` }}
                  />
                  <span className="num">{fmt(p.gradNorm, 3)}</span>
                </span>
                <span className="num">{fmt(ratio, 2)}</span>
              </div>
            );
          })}
        </div>
      </Card>
    </>
  );
}

function StepDetailView({ exp }: { exp: Experiment }) {
  const detail = useStore(exp.store, (s) => s.stepDetail)!;
  const params = Object.keys(detail.updates);
  const [param, setParam] = useState(params[0]);
  const before = detail.before;
  const pnode = before.nodes.find((n) => n.key === `param:${param}`);
  const after = detail.after.nodes.find((n) => n.key === `param:${param}`);
  const L0 = before.loss ?? NaN;
  const L1 = detail.after.loss ?? NaN;
  const upd = detail.updates[param];
  return (
    <Card
      title={`Step ${detail.record.step} in detail`}
      span={12}
      className="stepdetail"
      help="Δθ is the optimiser’s output, not −η∇θ, unless the optimiser is plain SGD without weight decay."
    >
      <ol className="pipeline" aria-label="One optimisation step">
        <li>
          <span className="pipe-k">Forward</span>
          <span className="num">L = {fmt(L0, 6)}</span>
          <span className="muted">batch of {exp.state.summary?.batchSize}</span>
        </li>
        <li>
          <span className="pipe-k">Backward</span>
          <span className="num">‖∇θ‖ = {fmt(detail.record.gradNorm, 4)}</span>
          <span className="muted">
            {detail.record.clipScale < 1
              ? `clipped ×${fmt(detail.record.clipScale, 3)}`
              : 'not clipped'}
          </span>
        </li>
        <li>
          <span className="pipe-k">Update</span>
          <span className="num">
            ‖Δθ‖ ={' '}
            {fmt(
              Math.sqrt(params.reduce((s, n) => s + detail.record.params[n].updateNorm ** 2, 0)),
              4,
            )}
          </span>
          <span className="muted">
            {exp.state.summary?.optimizer.kind}, η = {fmt(exp.state.summary?.optimizer.lr ?? 0)}
          </span>
        </li>
        <li>
          <span className="pipe-k">Forward again</span>
          <span className="num">L′ = {fmt(L1, 6)}</span>
          <span className={L1 < L0 ? 'muted' : 'is-bad'}>
            ΔL = {fmt(L1 - L0, 3)} on the same batch
          </span>
        </li>
      </ol>
      <div className="row">
        <label className="field-inline">
          <span className="field-label">Parameter</span>
          <select value={param} onChange={(e) => setParam(e.target.value)}>
            {params.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="step-tensors">
        {pnode && (
          <TensorView
            key={`a${param}`}
            tensor={{ shape: pnode.shape, data: pnode.value }}
            label="θ before"
            compact
            maxWidth={200}
          />
        )}
        {pnode?.grad && (
          <TensorView
            key={`b${param}`}
            tensor={{ shape: pnode.shape, data: pnode.grad }}
            label="∂L/∂θ"
            kind="grad"
            compact
            maxWidth={200}
          />
        )}
        {upd && (
          <TensorView
            key={`c${param}`}
            tensor={upd}
            label="Δθ applied"
            kind="update"
            compact
            maxWidth={200}
          />
        )}
        {after && (
          <TensorView
            key={`d${param}`}
            tensor={{ shape: after.shape, data: after.value }}
            label="θ after"
            compact
            maxWidth={200}
          />
        )}
      </div>
      <p className="field-hint">θ after = θ before + Δθ.</p>
    </Card>
  );
}
