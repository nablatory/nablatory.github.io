import { useEffect, useMemo, useRef, useState } from 'react';
import { FAMILIES, findModel, modelsOf } from '../../models/catalog';
import type { ModelDef } from '../../models/types';
import { Experiment } from '../experiment/controller';
import { useStore } from '../../state/store';
import { FamilyGlyph } from '../explore/glyphs';
import { Heatmap } from '../../viz/Heatmap';
import { ScaleBar } from '../../viz/ScaleBar';
import { LineChart } from '../../viz/LineChart';
import { symmetricScale } from '../../viz/colormap';
import { CLASS_COLORS } from '../../viz/views';
import { Button } from '../../ui/controls';
import { fmt, fmtMs, shortCite } from '../../ui/format';
import { href, modelPath, onLinkClick, preferredView, rememberView } from '../router';
import { FIDELITY } from './model/shared';

const HERO_STEPS = 300;
const HERO_CHUNK = 25;

/** The home page demonstration: a real MLP on two spirals, trained in this tab. */
function HeroLab() {
  const exp = useMemo(() => new Experiment('mlp'), []);
  const status = useStore(exp.store, (s) => s.status);
  const busy = useStore(exp.store, (s) => s.busy);
  const history = useStore(exp.store, (s) => s.history);
  const field = useStore(exp.store, (s) => s.procedures.field);
  const summary = useStore(exp.store, (s) => s.summary);
  const error = useStore(exp.store, (s) => s.error);
  const [running, setRunning] = useState(false);
  const stopRef = useRef(false);
  const seedRef = useRef(1);
  const plotRef = useRef<HTMLDivElement>(null);
  const [plotW, setPlotW] = useState(0);
  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setPlotW(el.clientWidth));
    ro.observe(el);
    setPlotW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let live = true;
    exp.open({ seed: 1 }).then(() => {
      if (live) exp.procedure('field', {});
    });
    return () => {
      live = false;
      stopRef.current = true;
      exp.close();
    };
  }, [exp]);

  const train = async () => {
    stopRef.current = false;
    setRunning(true);
    for (let done = 0; done < HERO_STEPS && !stopRef.current; done += HERO_CHUNK) {
      await exp.train(HERO_CHUNK);
      await exp.procedure('field', {});
      if (exp.state.error) break;
    }
    setRunning(false);
  };
  const step = async () => {
    await exp.step();
    await exp.procedure('field', {});
  };
  const reset = async () => {
    seedRef.current += 1;
    await exp.reinit(seedRef.current);
    await exp.procedure('field', {});
  };

  const last = history[history.length - 1];
  const val = [...history].reverse().find((r) => r.validation)?.validation;
  const lossPts = history
    .map((r) => ({ x: r.step, y: r.losses.main ?? Object.values(r.losses)[0] }))
    .filter((p) => Number.isFinite(p.y));
  const valPts = history
    .filter((r) => r.validation)
    .map((r) => ({ x: r.step, y: r.validation!.loss }));

  const t = field?.result.tensors;
  const S = Math.max(200, Math.min(400, plotW || 360));
  const ext = [-1.6, 1.6, -1.6, 1.6];
  const sx = (x: number) => ((x - ext[0]) / (ext[1] - ext[0])) * S;
  const sy = (y: number) => S - ((y - ext[2]) / (ext[3] - ext[2])) * S;
  const scale = t ? symmetricScale(t.field.data) : null;

  return (
    <figure className="hero-lab" aria-label="Live multilayer perceptron on two spirals">
      <div className="hero-plot-wrap" ref={plotRef}>
        <div className="hero-plot" style={{ width: S, height: S }}>
          {t && scale ? (
            <>
              <Heatmap
                data={t.field.data}
                rows={64}
                cols={64}
                scale={scale}
                cell={S / 64}
                label="Decision field z₁ − z₀ on a 64 × 64 grid"
                gridLines={false}
              />
              <svg width={S} height={S} className="hero-points" aria-hidden="true">
                {Array.from({ length: t.points.shape[0] }, (_, i) => (
                  <circle
                    key={i}
                    cx={sx(t.points.data[i * 2])}
                    cy={sy(t.points.data[i * 2 + 1])}
                    r={2.7}
                    fill={CLASS_COLORS[t.labels.data[i]]}
                    stroke="var(--surface)"
                    strokeWidth={0.8}
                  />
                ))}
              </svg>
            </>
          ) : (
            <div className="hero-wait">
              {status === 'error' ? 'The engine could not start.' : 'Starting the engine…'}
            </div>
          )}
        </div>
        {scale && <ScaleBar scale={scale} label="Logit margin z₁ − z₀" />}
      </div>
      <div className="hero-side">
        <dl className="readouts readouts-big hero-readout">
          <div>
            <dt>Step</dt>
            <dd className="num">{summary?.step ?? 0}</dd>
          </div>
          <div>
            <dt>Batch loss</dt>
            <dd className="num">
              {last ? fmt(last.losses.main ?? Object.values(last.losses)[0], 3) : '—'}
            </dd>
          </div>
          <div title="Held-out points, evaluated during training">
            <dt>Held-out accuracy</dt>
            <dd className="num">{val ? `${fmt(val.metrics.accuracy * 100, 3)} %` : '—'}</dd>
          </div>
        </dl>
        <div className="hero-chart">
          {lossPts.length > 1 ? (
            <LineChart
              series={[
                { id: 'loss', label: 'batch loss', color: 'var(--ink)', points: lossPts },
                ...(valPts.length
                  ? [
                      {
                        id: 'val',
                        label: 'held-out loss',
                        color: 'var(--cmp)',
                        points: valPts,
                        markers: true,
                      },
                    ]
                  : []),
              ]}
              height={132}
              logY
              xLabel="step"
            />
          ) : (
            <p className="muted hero-chart-empty">
              Press Train: {HERO_STEPS} optimisation steps run here, and the loss of every batch is
              drawn in this space.
            </p>
          )}
        </div>
        <div className="hero-actions">
          {running ? (
            <Button
              variant="primary"
              onClick={() => {
                stopRef.current = true;
                exp.stop();
              }}
            >
              Pause
            </Button>
          ) : (
            <Button variant="primary" disabled={!summary || !!busy} onClick={train}>
              Train
            </Button>
          )}
          <Button disabled={!summary || !!busy || running} onClick={step}>
            Step
          </Button>
          <Button variant="quiet" disabled={!summary || !!busy || running} onClick={reset}>
            New random weights
          </Button>
        </div>
        <figcaption className="hero-caption">
          A multilayer perceptron, three tanh layers of 16 units, learning to separate two spirals.
          Colour is the network’s output over the plane
          {field ? <>, computed in {fmtMs(field.ms)} on this device</> : null}.
        </figcaption>
        {error && (
          <p className="field-hint is-bad" role="alert">
            {error}
          </p>
        )}
      </div>
    </figure>
  );
}

function ModelCard({ m }: { m: ModelDef }) {
  const ref = m.references[0];
  const path = modelPath(m.id, preferredView());
  const fid = FIDELITY[m.fidelity];
  return (
    <li>
      <a className="model-card" href={href(path)} onClick={(e) => onLinkClick(e, path)}>
        <span className="mc-name">{m.name}</span>
        <span className="mc-summary">{m.summary}</span>
        <span className="mc-meta">
          {ref ? <span className="mc-cite">{shortCite(ref)}</span> : null}
          <span className={`tag tag-${m.fidelity}`} title={fid.title}>
            {fid.label}
          </span>
          {m.checkpoints?.length ? (
            <span className="tag tag-trained" title="Comes with weights trained in this engine">
              Trained weights
            </span>
          ) : null}
        </span>
      </a>
    </li>
  );
}

export function ExplorePage() {
  useEffect(() => {
    const id = location.hash.slice(1);
    if (id) document.getElementById(id)?.scrollIntoView();
  }, []);
  const total = FAMILIES.reduce((n, f) => n + modelsOf(f.id).length, 0);
  const hasMlp = !!findModel('mlp');
  const go = (view: 'guide' | 'lab') => (e: React.MouseEvent<HTMLAnchorElement>) => {
    rememberView(view);
    onLinkClick(e, modelPath('mlp', view));
  };
  return (
    <main className="page explore" id="main">
      <section className="hero">
        <div className="hero-text">
          <h1 className="hero-h">Neural networks you can open up</h1>
          <p className="hero-lede">
            {total} architectures, from the perceptron to world models, built from their published
            equations. They run and train in this browser tab, and every number on screen is
            computed here.
          </p>
          <div className="hero-cta">
            <a className="btn btn-primary btn-md" href={href('/m/mlp')} onClick={go('guide')}>
              Start with the guide
            </a>
            <a className="btn btn-plain btn-md" href={href('/m/mlp/lab')} onClick={go('lab')}>
              Open the lab
            </a>
          </div>
          <p className="hero-cta-note muted">
            Both open the network on this page. Every other model has the same two levels.
          </p>
        </div>
        {hasMlp && <HeroLab />}
        <ul className="hero-facts">
          <li>
            <strong>Computed on your device.</strong> Forward passes, gradients and updates run in a
            background thread of this page. Nothing you draw or type is sent anywhere.
          </li>
          <li>
            <strong>Two levels.</strong> The guide explains one model in plain words with a few
            controls. The lab shows every tensor, every gradient and every setting.
          </li>
          <li>
            <strong>Checked gradients.</strong> Backpropagation is compared with finite differences
            in float64; you can run the check for all {total} models on{' '}
            <a href={href('/about')} onClick={(e) => onLinkClick(e, '/about')}>
              How it works
            </a>
            .
          </li>
        </ul>
      </section>

      <section className="catalog" id="models" aria-labelledby="catalog-h">
        <nav className="family-index" aria-label="Families">
          <h2 id="catalog-h" className="family-index-h">
            {total} models
          </h2>
          <ul>
            {FAMILIES.map((f) => (
              <li key={f.id}>
                <a href={`#${f.id}`}>
                  {f.name}
                  <span className="muted num">{modelsOf(f.id).length}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <div className="families">
          {FAMILIES.map((f) => {
            const ms = modelsOf(f.id);
            return (
              <section key={f.id} id={f.id} className="family" aria-labelledby={`fam-${f.id}`}>
                <div className="family-head">
                  <FamilyGlyph family={f.id} />
                  <div>
                    <h3 id={`fam-${f.id}`} className="family-h">
                      {f.name}
                    </h3>
                    <p className="family-summary">{f.summary}</p>
                  </div>
                </div>
                <ul className="model-cards">
                  {ms.map((m) => (
                    <ModelCard key={m.id} m={m} />
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      </section>
    </main>
  );
}
