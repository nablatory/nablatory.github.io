import { useRef, useState, type ReactNode } from 'react';
import { MODELS } from '../../models/catalog';
import { engine } from '../../runtime/client';
import { createGpuMatmul } from '../../runtime/webgpu';
import { mmNN } from '../../engine/ops/linalg';
import { Rng } from '../../engine/rng';
import { Button } from '../../ui/controls';
import { fmtMs } from '../../ui/format';
import { href, modelPath, onLinkClick } from '../router';

type CheckRow = {
  id: string;
  name: string;
  maxRelError?: number;
  entries?: number;
  pass?: boolean;
  kinks?: number;
  params: string;
  ms?: number;
  error?: string;
};

/** Runs the float64 finite-difference gradient check of every model, here, now. */
function VerifyAll() {
  const [rows, setRows] = useState<CheckRow[]>([]);
  const [running, setRunning] = useState(false);
  const stop = useRef(false);
  const run = async () => {
    stop.current = false;
    setRunning(true);
    setRows([]);
    for (const m of MODELS) {
      if (stop.current) break;
      const session = `verify-${m.id}`;
      const t0 = performance.now();
      // Same parameters a model page opens with: the first checkpoint, else seeded initialisation.
      const ck = m.checkpoints?.[0];
      let row: CheckRow = { id: m.id, name: m.name, params: ck ? 'trained' : 'initial' };
      try {
        await engine.request({ type: 'open', session, model: m.id });
        const r = await engine.request({ type: 'gradcheck', session });
        row = {
          ...row,
          maxRelError: r.maxRelError,
          entries: r.entries.length,
          pass: r.pass,
          kinks: r.kinks,
          ms: performance.now() - t0,
        };
      } catch (e) {
        row = { ...row, error: e instanceof Error ? e.message : String(e) };
      } finally {
        engine.request({ type: 'close', session }).catch(() => undefined);
      }
      setRows((rs) => [...rs, row]);
    }
    setRunning(false);
  };
  const agree = rows.filter((r) => r.pass && !r.error).length;
  const done = !running && rows.length > 0;
  return (
    <div className="verify">
      <div className="row">
        {running ? (
          <Button onClick={() => (stop.current = true)}>Stop</Button>
        ) : (
          <Button variant="primary" onClick={run}>
            {rows.length ? 'Check again' : `Check all ${MODELS.length} models`}
          </Button>
        )}
        {running && (
          <span className="muted num" aria-live="polite">
            Checked {rows.length} of {MODELS.length}
          </span>
        )}
        {done && (
          <span
            className={`pill ${agree === rows.length ? 'pill-ok' : 'pill-bad'}`}
            aria-live="polite"
          >
            {agree} of {rows.length} agree
          </span>
        )}
      </div>
      {rows.length > 0 && (
        <div className="table-wrap">
          <table className="data-table verify-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Weights</th>
                <th scope="col" className="r">
                  Entries
                </th>
                <th scope="col" className="r">
                  Max. rel. error
                </th>
                <th scope="col">Result</th>
                <th scope="col" className="r">
                  Time
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>
                    <a
                      href={href(modelPath(r.id, 'lab'))}
                      onClick={(e) => onLinkClick(e, modelPath(r.id, 'lab'))}
                    >
                      {r.name}
                    </a>
                  </td>
                  <td className="muted">{r.params}</td>
                  <td className="r num">{r.entries ?? '—'}</td>
                  <td className="r num">
                    {r.maxRelError !== undefined ? r.maxRelError.toExponential(2) : '—'}
                  </td>
                  <td>
                    {r.error ? (
                      <span className="is-bad">{r.error}</span>
                    ) : (
                      <>
                        <span className={`pill ${r.pass ? 'pill-ok' : 'pill-bad'}`}>
                          {r.pass ? 'Agrees' : 'Exceeds tolerance'}
                        </span>
                        {r.kinks ? <span className="muted"> {r.kinks} at kinks</span> : null}
                      </>
                    )}
                  </td>
                  <td className="r num">{r.ms !== undefined ? fmtMs(r.ms) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

type BenchRow = { n: number; cpuMs: number; gpuMs?: number; maxAbs?: number };

function GpuBench() {
  const [state, setState] = useState<{
    status: 'idle' | 'running' | 'done' | 'unavailable';
    adapter?: string;
    rows: BenchRow[];
    error?: string;
  }>({ status: 'idle', rows: [] });
  const run = async () => {
    setState({ status: 'running', rows: [] });
    try {
      const gpu = await createGpuMatmul();
      const rows: BenchRow[] = [];
      const rng = new Rng(7);
      for (const n of [128, 256, 512]) {
        const a = Float32Array.from({ length: n * n }, () => rng.normal());
        const b = Float32Array.from({ length: n * n }, () => rng.normal());
        const c = new Float32Array(n * n);
        const t0 = performance.now();
        mmNN(a, 0, b, 0, c, 0, n, n, n);
        const row: BenchRow = { n, cpuMs: performance.now() - t0 };
        if (gpu) {
          await gpu.run(a, b, n, n, n); // warm-up: pipeline compilation is excluded from the timing
          const g = await gpu.run(a, b, n, n, n);
          let maxAbs = 0;
          for (let i = 0; i < c.length; i++) maxAbs = Math.max(maxAbs, Math.abs(c[i] - g.c[i]));
          row.gpuMs = g.ms;
          row.maxAbs = maxAbs;
        }
        rows.push(row);
        setState({ status: 'running', adapter: gpu?.adapter, rows: [...rows] });
      }
      gpu?.destroy();
      setState({ status: gpu ? 'done' : 'unavailable', adapter: gpu?.adapter, rows });
    } catch (e) {
      setState((s) => ({
        ...s,
        status: 'done',
        error: e instanceof Error ? e.message : String(e),
      }));
    }
  };
  return (
    <div className="verify">
      <div className="row">
        <Button onClick={run} disabled={state.status === 'running'}>
          Run benchmark
        </Button>
        {state.adapter && <span className="muted">{state.adapter}</span>}
        {state.status === 'unavailable' && (
          <span className="muted">WebGPU is not available here. Only the CPU was timed.</span>
        )}
      </div>
      {state.rows.length > 0 && (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col" className="r">
                  n × n
                </th>
                <th scope="col" className="r">
                  CPU
                </th>
                <th scope="col" className="r">
                  WebGPU
                </th>
                <th scope="col" className="r">
                  Max. difference
                </th>
              </tr>
            </thead>
            <tbody>
              {state.rows.map((r) => (
                <tr key={r.n}>
                  <td className="r num">{r.n}</td>
                  <td className="r num">{fmtMs(r.cpuMs)}</td>
                  <td className="r num">{r.gpuMs !== undefined ? fmtMs(r.gpuMs) : '—'}</td>
                  <td className="r num">
                    {r.maxAbs !== undefined ? r.maxAbs.toExponential(2) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {state.error && <p className="field-hint is-bad">{state.error}</p>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="about-row">
      <h2 className="about-h">{title}</h2>
      <div className="about-body">{children}</div>
    </section>
  );
}

export function AboutPage() {
  const licenses = new Map<string, string>();
  for (const m of MODELS)
    for (const d of m.datasets) if (d.license) licenses.set(d.name, d.license);
  const count = (f: string) => MODELS.filter((m) => m.fidelity === f).length;
  return (
    <main className="page about" id="main">
      <header className="page-head">
        <h1 className="page-h">How it works</h1>
        <p className="page-lede">
          One small autodiff engine, running in your browser, computes every number on this site.
        </p>
      </header>

      <Section title="Computed on your device">
        <p>
          Each model is a sequence of tensor operations recorded on a tape. The backward pass gives
          the gradient of the loss for every intermediate tensor, not only for the weights.
        </p>
        <p>
          The engine runs in a Web Worker in 32-bit floats; gradient checks use 64-bit. Nothing is
          uploaded, and saved experiments stay in this browser.
        </p>
      </Section>

      <Section title="Published size, equations or reduced">
        <ul className="about-list">
          <li>
            <span className="tag tag-exact">Published size</span> {count('exact')} of{' '}
            {MODELS.length}: the architecture at the size given in the paper.
          </li>
          <li>
            <span className="tag tag-equations">Published equations</span> {count('equations')}: the
            paper defines the equations but no fixed size, so the size here is a choice.
          </li>
          <li>
            <span className="tag tag-reduced">Reduced scale</span> {count('reduced')}: the published
            equations with fewer layers, channels, tokens or steps, so that they train in a browser.
          </li>
        </ul>
        <p>
          The Notes tab of each model&rsquo;s lab says what was reduced and what the implementation
          leaves out.
        </p>
      </Section>

      <Section title="Checked against references">
        <ul className="about-list">
          <li>
            45 operations and their gradients agree with PyTorch 2.8.0 in float64, to a scaled error
            below 10⁻¹⁰; the six optimisers follow PyTorch&rsquo;s update rules step by step.
          </li>
          <li>
            Two composite computations, a small convolutional network and an attention block, agree
            with TensorFlow.js.
          </li>
          <li>Every model&rsquo;s backward pass is compared with finite differences.</li>
        </ul>
        <p>
          The first two run in the test suite. Run the third here, on every model: each opens with
          the weights its page uses, a sample of parameter entries is perturbed by ±ε in float64,
          and (L(θ+ε) − L(θ−ε)) / 2ε is compared with the gradient from the tape. Models trained in
          several phases, such as a GAN&rsquo;s discriminator and generator, have each phase&rsquo;s
          loss checked on the parameters that phase updates. Entries where a ReLU or max-pool
          switches branch within ±ε, so that the two one-sided differences disagree, are counted as
          kinks and left out of the maximum error.
        </p>
        <VerifyAll />
      </Section>

      <Section title="Trained weights">
        <p>
          Trained weights were produced with this same engine by scripts in the repository. Each
          file records its dataset, seed, steps, optimiser and final metrics.
        </p>
      </Section>

      <Section title="WebGPU">
        <p>
          Training runs on the CPU, because the tape is synchronous. This benchmark times a WebGPU
          matrix multiply against the CPU kernel and checks that they agree.
        </p>
        <GpuBench />
      </Section>

      <Section title="Data">
        <p>
          Synthetic data is generated from fixed seeds. Real datasets are small shipped subsets.
        </p>
        <dl className="about-licences">
          {[...licenses].map(([name, lic]) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd>{lic}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section title="Source">
        <p>
          Open source under the MIT licence. New models must pass a gradient check, shape and
          parameter-count tests, and a training run that lowers the loss.
        </p>
      </Section>
    </main>
  );
}
