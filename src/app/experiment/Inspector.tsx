import { useEffect, useMemo, useState, type ReactNode } from 'react';
import type { ModelDef } from '../../models/types';
import type { Trace, TraceNode } from '../../runtime/trace';
import { useStore } from '../../state/store';
import type { Experiment } from './controller';
import { buildModules, describeModule, type ModuleGraph } from '../../viz/architecture';
import { opMath } from '../../viz/opMath';
import { TensorView, flatToIndex } from '../../viz/TensorView';
import { Tex } from '../../ui/Tex';
import { Button, Card, Disclosure, NumberInput, Provenance, Readout, Tip } from '../../ui/controls';
import { fmt, fmtCount, fmtMs, fmtShape, metricLabel } from '../../ui/format';
import { Term, termAttrs } from '../../ui/Term';
import { GLOSSARY, findTerm, opTerm } from '../../ui/glossary';

/** 1e-6 → 10⁻⁶; other values in plain exponent notation. */
function fmtPow10(v: number): string {
  const e = Math.log10(v);
  if (Number.isInteger(e))
    return `10${String(e)
      .replace('-', '⁻')
      .replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`;
  return v.toExponential(0);
}

function l2(d?: ArrayLike<number>): number {
  if (!d) return 0;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

export function mathFor(model: ModelDef, scope: string) {
  const parts = scope.split('/');
  for (let i = parts.length; i > 0; i--) {
    const key = parts.slice(0, i).join('/');
    // Exact scope first, then its numbered-stem ("block3" → "block").
    const m =
      model.math[key] ??
      model.math[key.replace(/\d+$/, '')] ??
      model.math[key.replace(/\d+(?=\/|$)/g, '')];
    if (m) return m;
  }
  return undefined;
}

export function Inspector({
  exp,
  model,
  graph,
}: {
  exp: Experiment;
  model: ModelDef;
  graph: ModuleGraph | null;
}) {
  const trace = useStore(exp.store, (s) => s.trace);
  const previous = useStore(exp.store, (s) => s.previous);
  const selection = useStore(exp.store, (s) => s.selection);
  const summary = useStore(exp.store, (s) => s.summary);
  if (!trace || !summary || !selection) return null;
  const prevByKey = previous ? new Map(previous.nodes.map((n) => [n.key, n])) : null;

  let body;
  if (selection?.kind === 'node') {
    const node = trace.nodes.find((n) => n.id === selection.id);
    body = node ? (
      <NodeInspector
        exp={exp}
        model={model}
        trace={trace}
        node={node}
        prev={prevByKey?.get(node.key) ?? null}
      />
    ) : null;
  } else if (selection?.kind === 'module' && graph) {
    const m = graph.byName.get(selection.name);
    body = m ? (
      <section className="insp-section">
        <header className="insp-head">
          <p className="insp-kicker">
            <Term id={m.category}>{GLOSSARY[m.category]?.term ?? 'Block'}</Term>
          </p>
          <h2>{m.label}</h2>
          <p className="muted">
            {describeModule(m)} · output {fmtShape(m.output.shape)}
            {m.paramCount ? ` · ${fmtCount(m.paramCount)} parameters` : ''}
          </p>
        </header>
        <MathBlock model={model} scope={m.output.scope || m.name} />
        <div className="insp-sub">
          <h3>Operations</h3>
          <ol className="op-list">
            {m.ops.map((n) => (
              <li key={n.id}>
                <button
                  type="button"
                  onClick={() => exp.select({ kind: 'node', id: n.id })}
                  {...termAttrs(opTerm(n.op), { title: n.op })}
                >
                  <span>{n.op}</span>
                  <span className="muted">{fmtShape(n.shape)}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
        {m.params.length > 0 && (
          <div className="insp-sub">
            <h3>Parameters</h3>
            <table className="param-table">
              <thead>
                <tr>
                  <th>name</th>
                  <th>shape</th>
                  <th>
                    <Term id="parameters" note="Euclidean norm of the parameter tensor.">
                      ‖θ‖
                    </Term>
                  </th>
                  <th>
                    <Term
                      id="gradient"
                      note="Norm of the loss gradient with respect to this parameter tensor."
                    >
                      ‖∂L/∂θ‖
                    </Term>
                  </th>
                </tr>
              </thead>
              <tbody>
                {m.params.map((p) => (
                  <tr key={p.id}>
                    <td>
                      <button
                        type="button"
                        className="linkish"
                        onClick={() => exp.select({ kind: 'node', id: p.id })}
                      >
                        {p.param}
                      </button>
                    </td>
                    <td>{p.shape.join('×')}</td>
                    <td className="num">{fmt(l2(p.value), 3)}</td>
                    <td className="num grad-num">{p.grad ? fmt(l2(p.grad), 3) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="insp-sub">
          <h3>Output</h3>
          <TensorView
            key={`v${m.output.key}`}
            tensor={{ shape: m.output.shape, data: m.output.value }}
            label="value"
            previous={prevOf(prevByKey, m.output, 'value')}
          />
          {m.output.grad && (
            <TensorView
              key={`g${m.output.key}`}
              tensor={{ shape: m.output.shape, data: m.output.grad }}
              label="∂L/∂y"
              kind="grad"
              previous={prevOf(prevByKey, m.output, 'grad')}
            />
          )}
        </div>
      </section>
    ) : null;
  }
  return (
    <div className="inspector">
      <div className="insp-nav">
        <button type="button" className="insp-back" onClick={() => exp.select(null)}>
          <svg viewBox="0 0 8 8" width="8" height="8" aria-hidden="true">
            <path d="M5.5 1L2 4l3.5 3" fill="none" stroke="currentColor" strokeWidth="1.3" />
          </svg>
          Overview
        </button>
        {selection.kind === 'node' && graph?.moduleOf.get(selection.id) && (
          <button
            type="button"
            className="insp-back"
            onClick={() => exp.select({ kind: 'module', name: graph.moduleOf.get(selection.id)! })}
          >
            Block {graph.byName.get(graph.moduleOf.get(selection.id)!)?.label}
          </button>
        )}
      </div>
      {body}
    </div>
  );
}

function prevOf(map: Map<string, TraceNode> | null, n: TraceNode, field: 'value' | 'grad') {
  const p = map?.get(n.key);
  const d = p?.[field];
  return d && d.length === n.value.length ? { shape: p!.shape, data: d } : null;
}

function MathBlock({ model, scope }: { model: ModelDef; scope: string }) {
  const m = mathFor(model, scope);
  if (!m) return null;
  return (
    <div className="math-block">
      <Tex tex={m.tex} block />
      {m.text && <p className="prose-small">{m.text}</p>}
    </div>
  );
}

/** Loss, metrics and sizes of the current computation, above the graph. */
export function NetworkSummary({
  exp,
  trace,
  className = 'net-summary',
}: {
  exp: Experiment;
  trace: Trace;
  className?: string;
}) {
  const summary = useStore(exp.store, (s) => s.summary)!;
  const previous = useStore(exp.store, (s) => s.previous);
  const flops = trace.nodes.reduce((s, n) => s + n.flops, 0);
  const lossPrev = previous?.loss;
  const items: { label: ReactNode; value: ReactNode; title?: string }[] = [];
  if (trace.loss !== undefined)
    items.push({
      label: <Term id="loss">Loss</Term>,
      value: (
        <>
          {fmt(trace.loss, 5)}
          {lossPrev !== undefined && lossPrev !== trace.loss && (
            <span className={`delta ${trace.loss < lossPrev ? 'is-down' : 'is-up'}`}>
              {trace.loss < lossPrev ? '−' : '+'}
              {fmt(Math.abs(trace.loss - lossPrev), 3)}
            </span>
          )}
        </>
      ),
      title: lossPrev !== undefined ? `Previous computation: ${fmt(lossPrev, 6)}` : undefined,
    });
  for (const [k, v] of Object.entries(trace.metrics))
    items.push({
      label: <Term id={findTerm(metricLabel(k))}>{metricLabel(k)}</Term>,
      value: fmt(v, 4),
    });
  items.push(
    {
      label: (
        <>
          <Term id="parameters">Parameters</Term> <Provenance kind="analytic" />
        </>
      ),
      value: fmtCount(summary.paramCount),
    },
    {
      label: (
        <>
          <Term id="flops">FLOPs per forward pass</Term> <Provenance kind="analytic" />
        </>
      ),
      value: fmtCount(flops),
    },
    {
      label: (
        <>
          <Term id="timing">Forward, backward</Term> <Provenance kind="measured" />
        </>
      ),
      value: `${fmtMs(trace.timing.forwardMs)}, ${trace.objective ? fmtMs(trace.timing.backwardMs) : '—'}`,
    },
  );
  return <Readout items={items} className={className} />;
}

/** Backpropagated gradients against central finite differences, on demand. */
export function GradientCheckCard({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const eps = model.gradCheck?.eps ?? 1e-6;
  const tol = model.gradCheck?.tol ?? 1e-5;
  const gradcheck = useStore(exp.store, (s) => s.gradcheck);
  const busy = useStore(exp.store, (s) => s.busy);
  // Disagreeing entries first, then kinks, then the rest in sampling order.
  const rank = (e: { pass: boolean; kink: boolean }) => (e.pass ? 2 : e.kink ? 1 : 0);
  const shown = gradcheck
    ? [...gradcheck.entries].sort((a, b) => rank(a) - rank(b)).slice(0, 24)
    : [];
  return (
    <Card
      title="Gradient check"
      help={
        <>
          Backpropagated gradients are compared with central finite differences (L(θ+ε) − L(θ−ε))/2ε
          in float64, ε = {fmtPow10(eps)}, on a small training batch. An entry agrees when its
          relative error is below {fmtPow10(tol)}.
          {(model.training.phases?.length ?? 0) > 1 &&
            ` Each training phase (${model.training.phases!.map((p) => p.label).join('; ')}) is checked on the parameters it updates.`}
        </>
      }
      span={6}
      actions={
        <Button size="sm" disabled={!!busy} onClick={() => exp.gradcheck()}>
          {gradcheck ? 'Check again' : 'Check gradients'}
        </Button>
      }
    >
      {!gradcheck ? (
        <p className="muted">
          Compares every sampled gradient entry with a finite difference computed in float64.
        </p>
      ) : (
        <>
          <p className={gradcheck.pass ? 'check-pass' : 'check-fail'}>
            <span className={`pill ${gradcheck.pass ? 'pill-ok' : 'pill-bad'}`}>
              {gradcheck.pass ? 'Agrees' : 'Exceeds tolerance'}
            </span>{' '}
            max relative error {fmt(gradcheck.maxRelError, 3)} over{' '}
            {gradcheck.entries.length - gradcheck.kinks} entries
            {gradcheck.kinks > 0 &&
              `; ${gradcheck.kinks} ${gradcheck.kinks === 1 ? 'entry sits' : 'entries sit'} at a kink, where the one-sided differences disagree and backpropagation returns one of them`}
          </p>
          {!gradcheck.pass && (
            <p className="field-hint">
              Central differences assume the loss is smooth within ±ε. If a disagreeing entry sits
              where a ReLU, leaky ReLU or max-pool switches branch (common right after
              initialisation), the finite difference straddles the kink and is not a valid reference
              there; re-run after a few training steps to sample other points.
            </p>
          )}
          <div className="table-wrap">
            <table className="param-table">
              <thead>
                <tr>
                  <th>parameter[i]</th>
                  <th>autodiff</th>
                  <th>finite diff.</th>
                  <th>rel. err</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((e, i) => (
                  <tr
                    key={i}
                    className={e.pass ? '' : e.kink ? 'is-kink' : 'is-bad'}
                    title={
                      e.kink
                        ? `Kink: one-sided differences ${fmt(e.forward ?? 0, 5)} (right) and ${fmt(e.backward ?? 0, 5)} (left)`
                        : undefined
                    }
                  >
                    <td className="mono">
                      {e.param}[{e.index}]
                    </td>
                    <td className="num">{fmt(e.analytic, 5)}</td>
                    <td className="num">{fmt(e.numeric, 5)}</td>
                    <td className="num">{e.kink ? 'kink' : fmt(e.relError, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {shown.length < gradcheck.entries.length && (
            <p className="field-hint">
              {shown.length} of {gradcheck.entries.length} entries shown, disagreeing ones first.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function NodeInspector({
  exp,
  model,
  trace,
  node,
  prev,
}: {
  exp: Experiment;
  model: ModelDef;
  trace: Trace;
  node: TraceNode;
  prev: TraceNode | null;
}) {
  const vjp = useStore(exp.store, (s) => s.vjp);
  const summary = useStore(exp.store, (s) => s.summary)!;
  const [editing, setEditing] = useState<{ flat: number; value: number } | null>(null);
  const math = opMath(node);
  const isParam = node.op === 'param';
  const byId = useMemo(() => new Map(trace.nodes.map((n) => [n.id, n])), [trace]);
  const parents = node.parents.map((id) => byId.get(id)).filter(Boolean) as TraceNode[];
  const paramInfo = isParam ? summary.params.find((p) => p.name === node.param) : undefined;
  const override = summary.overrides.find((o) => o.key === node.key);
  const marked = useMemo(() => {
    const s = new Set<number>();
    if (isParam) for (const e of summary.edits) if (e.param === node.param) s.add(e.index);
    if (override) for (const e of override.edits) s.add(e.index);
    return s;
  }, [isParam, summary.edits, node.param, override]);

  useEffect(() => {
    setEditing(null);
    if (!isParam && node.grad && node.op !== 'input') exp.vjp(node.key);
  }, [node.key, node.grad, isParam, node.op, exp]);

  const apply = (value: number) => {
    if (!editing) return;
    if (isParam) exp.editParam(node.param!, editing.flat, value);
    else {
      const edits = (override?.edits ?? []).filter((e) => e.index !== editing.flat);
      exp.setOverride(node.key, [...edits, { index: editing.flat, value }]);
    }
    setEditing(null);
  };

  return (
    <section className="insp-section">
      <header className="insp-head">
        <p className="insp-kicker">
          {isParam ? 'Parameter' : node.op === 'input' ? 'Input' : 'Operation'}
        </p>
        <h2>{isParam ? node.param : node.op}</h2>
        <p className="muted">
          {node.key} · {fmtShape(node.shape)}
          {node.edited && ' · patched'}
        </p>
        {paramInfo && (
          <p className="muted">
            {paramInfo.trainable
              ? paramInfo.frozen
                ? 'frozen'
                : 'trainable'
              : 'fixed (not trained)'}{' '}
            · init {paramInfo.init} · values: {paramInfo.source}
          </p>
        )}
      </header>
      {math && (
        <div className="math-block">
          <Tex tex={math.forward} block />
        </div>
      )}
      <MathBlock model={model} scope={node.scope} />
      {Object.keys(node.attrs).length > 0 && (
        <dl className="kv kv-compact">
          {Object.entries(node.attrs).map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{Array.isArray(v) ? v.join(', ') : String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="insp-sub">
        <h3>{isParam ? 'Values' : 'Output'}</h3>
        <TensorView
          key={`v${node.key}`}
          tensor={{ shape: node.shape, data: node.value }}
          label={isParam ? 'θ' : 'y'}
          previous={
            prev && prev.value.length === node.value.length
              ? { shape: prev.shape, data: prev.value }
              : null
          }
          marked={marked}
          onPick={node.op === 'const' ? undefined : (flat, value) => setEditing({ flat, value })}
        />
        {editing && (
          <div className="edit-row">
            <span className="num">[{flatToIndex(node.shape, editing.flat).join(', ')}]</span>
            <span className="muted">{fmt(editing.value, 6)} →</span>
            <NumberInput value={editing.value} onCommit={apply} ariaLabel="New value" width={110} />
            <Button size="sm" variant="quiet" onClick={() => setEditing(null)}>
              Cancel
            </Button>
            <p className="field-hint">
              {isParam
                ? 'Sets this parameter; every downstream value is recomputed.'
                : 'Patches this activation: later operations use the new value; no gradient flows through the patched element.'}
            </p>
          </div>
        )}
      </div>
      {node.grad && (
        <div className="insp-sub">
          <h3>
            Gradient {trace.objective && <span className="muted">of {trace.objective.label}</span>}
          </h3>
          <TensorView
            key={`g${node.key}`}
            tensor={{ shape: node.shape, data: node.grad }}
            label={isParam ? '∂L/∂θ' : '∂L/∂y'}
            kind="grad"
            previous={
              prev?.grad && prev.grad.length === node.grad.length
                ? { shape: prev.shape, data: prev.grad }
                : null
            }
          />
        </div>
      )}
      {math && node.grad && vjp?.key === node.key && vjp.result && (
        <div className="insp-sub">
          <h3 className="insp-h-tip">
            Chain rule here
            <Tip>
              The upstream gradient ḡ = ∂L/∂y is multiplied by this operation’s local Jacobian. Each
              panel shows what this operation alone sends to one input; an input used elsewhere
              receives the sum of all such contributions.
            </Tip>
          </h3>
          {vjp.result.parents.map((p, i) => {
            const parent = parents[i];
            if (!p.contribution || !parent) return null;
            const isTotal =
              p.total &&
              l2(p.total.data) > 0 &&
              Math.abs(l2(p.total.data) - l2(p.contribution.data)) < 1e-9 * (1 + l2(p.total.data));
            return (
              <div key={i} className="vjp">
                <div className="vjp-eq">
                  <Tex tex={math.backward[i] ?? ''} />
                  <span className="muted">
                    {' '}
                    → {math.inputs[i] ?? `input ${i}`} ={' '}
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => exp.select({ kind: 'node', id: parent.id })}
                    >
                      {parent.param ?? parent.key}
                    </button>
                  </span>
                </div>
                <TensorView
                  tensor={p.contribution}
                  label={`contribution to ∂L/∂${math.inputs[i] ?? 'x'}`}
                  kind="grad"
                  compact
                  maxWidth={280}
                />
                <p className="field-hint">
                  {isTotal
                    ? 'Equal to the total gradient of this input (used once).'
                    : 'Part of the total gradient of this input (used more than once).'}
                </p>
              </div>
            );
          })}
        </div>
      )}
      {node.aux && (
        <Disclosure title="Saved intermediate values" meta={Object.keys(node.aux).join(', ')}>
          {Object.entries(node.aux).map(([k, t]) => (
            <TensorView key={k} tensor={t} label={k} compact maxWidth={280} />
          ))}
        </Disclosure>
      )}
      {node.op !== 'param' &&
        node.op !== 'input' &&
        trace.outputs &&
        Object.entries(trace.outputs).some(([, id]) => id === node.id) && (
          <OutputGradTarget exp={exp} node={node} trace={trace} />
        )}
      {parents.length > 0 && (
        <div className="insp-sub">
          <h3>Inputs</h3>
          <ol className="op-list">
            {parents.map((p) => (
              <li key={p.id}>
                <button type="button" onClick={() => exp.select({ kind: 'node', id: p.id })}>
                  <span>{p.param ?? p.key}</span>
                  <span className="muted">{fmtShape(p.shape)}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function OutputGradTarget({
  exp,
  node,
  trace,
}: {
  exp: Experiment;
  node: TraceNode;
  trace: Trace;
}) {
  const name = Object.entries(trace.outputs).find(([, id]) => id === node.id)![0];
  const [index, setIndex] = useState(0);
  return (
    <div className="insp-sub">
      <h3 className="insp-h-tip">
        Differentiate this output
        <Tip>
          Backpropagate from one element of “{name}” instead of the loss, e.g. a class logit, to see
          which inputs and weights influence it.
        </Tip>
      </h3>
      <div className="row">
        <NumberInput
          value={index}
          integer
          min={0}
          max={node.value.length - 1}
          onCommit={setIndex}
          width={72}
          ariaLabel="Element index"
        />
        <Button size="sm" onClick={() => exp.inspect({ kind: 'output', name, index })}>
          ∂{name}[{index}]
        </Button>
        <Button size="sm" variant="quiet" onClick={() => exp.inspect({ kind: 'loss' })}>
          Back to loss
        </Button>
      </div>
    </div>
  );
}

export function useModuleGraph(trace: Trace | null): ModuleGraph | null {
  return useMemo(() => (trace ? buildModules(trace) : null), [trace]);
}
