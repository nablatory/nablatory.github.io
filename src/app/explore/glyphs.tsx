import type { FamilyId } from '../../models/types';

/**
 * Structural diagrams of each family's defining computation. They depict
 * connectivity only; no values are drawn.
 */
const S = { fill: 'none', stroke: 'var(--ink-2)', strokeWidth: 1.1 } as const;
const node = (x: number, y: number, r = 3.2, k?: string) => (
  <circle
    key={k ?? `${x},${y}`}
    cx={x}
    cy={y}
    r={r}
    fill="var(--plot)"
    stroke="var(--ink-2)"
    strokeWidth={1.1}
  />
);

function Feedforward() {
  const cols = [
    [18, 30, 42, 54].map((y) => [20, y + 4]),
    [12, 26, 40, 54, 68].map((y) => [66, y]),
    [28, 52].map((y) => [112, y]),
  ];
  const lines = [];
  for (let c = 0; c < 2; c++)
    for (const a of cols[c])
      for (const b of cols[c + 1])
        lines.push(
          <line
            key={`${a}${b}`}
            x1={a[0]}
            y1={a[1]}
            x2={b[0]}
            y2={b[1]}
            stroke="var(--rule-strong)"
            strokeWidth={0.7}
          />,
        );
  return (
    <>
      {lines}
      {cols.flat().map(([x, y]) => node(x, y))}
    </>
  );
}

function Convolutional() {
  const cells = [];
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++)
      cells.push(
        <rect
          key={`${i}${j}`}
          x={8 + j * 9}
          y={11 + i * 9}
          width={9}
          height={9}
          {...S}
          stroke="var(--rule-strong)"
          strokeWidth={0.6}
        />,
      );
  const out = [];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      out.push(
        <rect
          key={`o${i}${j}`}
          x={90 + j * 9}
          y={20 + i * 9}
          width={9}
          height={9}
          {...S}
          stroke="var(--rule-strong)"
          strokeWidth={0.6}
        />,
      );
  return (
    <>
      {cells}
      <rect
        x={17}
        y={20}
        width={27}
        height={27}
        fill="var(--mark)"
        stroke="var(--ink)"
        strokeWidth={1.3}
      />
      {out}
      <rect
        x={99}
        y={29}
        width={9}
        height={9}
        fill="var(--mark)"
        stroke="var(--ink)"
        strokeWidth={1.3}
      />
      <path
        d="M44 20 L99 29 M44 47 L99 38"
        stroke="var(--ink-3)"
        strokeWidth={0.8}
        strokeDasharray="2 2"
      />
    </>
  );
}

function Recurrent() {
  const xs = [18, 50, 82, 114];
  return (
    <>
      {xs.map((x, i) => (
        <g key={x}>
          <rect x={x - 10} y={28} width={20} height={18} rx={2} {...S} />
          <line x1={x} y1={66} x2={x} y2={47} {...S} markerEnd="url(#g-arrow)" />
          <circle cx={x} cy={68} r={2.6} fill="var(--ink-2)" />
          {i < 3 && (
            <line x1={x + 10} y1={37} x2={x + 22} y2={37} {...S} markerEnd="url(#g-arrow)" />
          )}
        </g>
      ))}
      <text x={6} y={18} className="glyph-text">
        h₀ → h₁ → h₂ → h₃
      </text>
    </>
  );
}

function Transformer() {
  const n = 6;
  const cells = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      cells.push(
        <rect
          key={`${i}${j}`}
          x={66 + j * 9}
          y={10 + i * 9}
          width={9}
          height={9}
          fill={j <= i ? 'var(--tint-2)' : 'none'}
          stroke="var(--rule-strong)"
          strokeWidth={0.6}
        />,
      );
  return (
    <>
      {Array.from({ length: n }, (_, i) => (
        <rect
          key={i}
          x={8}
          y={10 + i * 9}
          width={34}
          height={7}
          rx={1}
          fill="var(--tint)"
          stroke="var(--ink-3)"
          strokeWidth={0.7}
        />
      ))}
      <text x={46} y={40} className="glyph-text">
        QKᵀ
      </text>
      {cells}
    </>
  );
}

function Autoencoder() {
  return (
    <>
      <path d="M10 10 L56 30 L56 46 L10 66 Z" {...S} fill="var(--tint)" />
      <path d="M122 10 L76 30 L76 46 L122 66 Z" {...S} fill="var(--tint)" />
      <rect
        x={60}
        y={31}
        width={12}
        height={14}
        fill="var(--mark)"
        stroke="var(--ink)"
        strokeWidth={1.2}
      />
      <text x={66} y={60} textAnchor="middle" className="glyph-text">
        z
      </text>
    </>
  );
}

function Diffusion() {
  const xs = [8, 34, 60, 86, 112];
  let seed = 3;
  const rnd = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
  return (
    <>
      {xs.map((x, k) => (
        <g key={x}>
          <rect x={x} y={22} width={18} height={18} {...S} />
          {k === 0 && (
            <circle cx={x + 9} cy={31} r={5} fill="none" stroke="var(--ink)" strokeWidth={1.4} />
          )}
          {Array.from({ length: k * 9 }, (_, i) => (
            <circle
              key={i}
              cx={x + 2 + rnd() * 14}
              cy={24 + rnd() * 14}
              r={0.8}
              fill="var(--ink-2)"
            />
          ))}
        </g>
      ))}
      <path d="M20 50 C50 62, 90 62, 120 50" {...S} markerEnd="url(#g-arrow)" />
      <path d="M120 14 C90 4, 50 4, 22 14" {...S} strokeDasharray="3 2" markerEnd="url(#g-arrow)" />
      <text x={70} y={72} textAnchor="middle" className="glyph-text">
        q(xₜ | xₜ₋₁)
      </text>
    </>
  );
}

function Graph() {
  const pts: [number, number][] = [
    [66, 38],
    [30, 18],
    [28, 58],
    [102, 16],
    [106, 58],
    [66, 70],
  ];
  const edges: [number, number][] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [3, 4],
    [4, 5],
    [2, 5],
  ];
  return (
    <>
      {edges.map(([a, b]) => (
        <line
          key={`${a}${b}`}
          x1={pts[a][0]}
          y1={pts[a][1]}
          x2={pts[b][0]}
          y2={pts[b][1]}
          stroke="var(--rule-strong)"
          strokeWidth={0.9}
        />
      ))}
      {[1, 2, 3, 4].map((k) => {
        const [x, y] = pts[k];
        const [cx, cy] = pts[0];
        return (
          <line
            key={k}
            x1={x + (cx - x) * 0.25}
            y1={y + (cy - y) * 0.25}
            x2={x + (cx - x) * 0.72}
            y2={y + (cy - y) * 0.72}
            stroke="var(--ink)"
            strokeWidth={1.1}
            markerEnd="url(#g-arrow)"
          />
        );
      })}
      {pts.map(([x, y], i) => node(x, y, i === 0 ? 5 : 3.6, String(i)))}
    </>
  );
}

function Gan() {
  return (
    <>
      <text x={8} y={42} className="glyph-text">
        z
      </text>
      <line x1={16} y1={38} x2={30} y2={38} {...S} markerEnd="url(#g-arrow)" />
      <rect x={32} y={28} width={22} height={20} rx={2} {...S} />
      <text x={43} y={42} textAnchor="middle" className="glyph-text">
        G
      </text>
      <line x1={54} y1={38} x2={78} y2={38} {...S} markerEnd="url(#g-arrow)" />
      <text x={66} y={32} textAnchor="middle" className="glyph-text">
        x̃
      </text>
      <text x={66} y={68} textAnchor="middle" className="glyph-text">
        x
      </text>
      <line x1={70} y1={62} x2={82} y2={46} {...S} markerEnd="url(#g-arrow)" />
      <rect x={82} y={28} width={22} height={20} rx={2} {...S} />
      <text x={93} y={42} textAnchor="middle" className="glyph-text">
        D
      </text>
      <line x1={104} y1={38} x2={122} y2={38} {...S} markerEnd="url(#g-arrow)" />
      <path
        d="M93 28 C93 8, 43 8, 43 26"
        {...S}
        strokeDasharray="3 2"
        stroke="var(--grad)"
        markerEnd="url(#g-arrow)"
      />
    </>
  );
}

function WorldModel() {
  const xs = [14, 48, 82, 116];
  return (
    <>
      {xs.map((x, i) => (
        <g key={x}>
          {node(x, 44, 6)}
          <rect
            x={x - 7}
            y={10}
            width={14}
            height={12}
            {...S}
            strokeDasharray={i > 1 ? '2 2' : undefined}
          />
          <line x1={x} y1={37} x2={x} y2={23} {...S} />
          {i < 3 && (
            <line
              x1={x + 7}
              y1={44}
              x2={x + 26}
              y2={44}
              {...S}
              strokeDasharray={i > 0 ? '3 2' : undefined}
              markerEnd="url(#g-arrow)"
            />
          )}
          {i < 3 && (
            <text x={x + 17} y={64} textAnchor="middle" className="glyph-text">
              a
            </text>
          )}
          {i < 3 && (
            <line x1={x + 17} y1={57} x2={x + 17} y2={47} stroke="var(--ink-3)" strokeWidth={0.8} />
          )}
        </g>
      ))}
    </>
  );
}

function Ssm() {
  return (
    <>
      <text x={8} y={42} className="glyph-text">
        xₜ
      </text>
      <line x1={22} y1={38} x2={44} y2={38} {...S} markerEnd="url(#g-arrow)" />
      <text x={33} y={32} textAnchor="middle" className="glyph-text">
        B̄
      </text>
      <rect x={46} y={26} width={36} height={24} rx={2} {...S} />
      <text x={64} y={42} textAnchor="middle" className="glyph-text">
        hₜ
      </text>
      <path d="M58 26 C52 6, 76 6, 70 26" {...S} markerEnd="url(#g-arrow)" />
      <text x={64} y={10} textAnchor="middle" className="glyph-text">
        Ā
      </text>
      <line x1={82} y1={38} x2={106} y2={38} {...S} markerEnd="url(#g-arrow)" />
      <text x={94} y={32} textAnchor="middle" className="glyph-text">
        C
      </text>
      <text x={110} y={42} className="glyph-text">
        yₜ
      </text>
    </>
  );
}

function NeuralDynamics() {
  const arrows = [];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 6; j++) {
      const x = 14 + j * 20;
      const y = 12 + i * 18;
      const a = Math.atan2(-(x - 66) * 0.6, (y - 40) * 1.2 + 6);
      arrows.push(
        <line
          key={`${i}${j}`}
          x1={x}
          y1={y}
          x2={x + 6 * Math.cos(a)}
          y2={y + 6 * Math.sin(a)}
          stroke="var(--rule-strong)"
          strokeWidth={0.8}
        />,
      );
    }
  return (
    <>
      {arrows}
      <path
        d="M20 60 C40 72, 100 70, 112 44 S 70 8, 44 24"
        fill="none"
        stroke="var(--ink)"
        strokeWidth={1.3}
        markerEnd="url(#g-arrow)"
      />
      {node(20, 60, 2.6)}
    </>
  );
}

const GLYPHS: Record<FamilyId, () => React.ReactElement> = {
  feedforward: Feedforward,
  convolutional: Convolutional,
  recurrent: Recurrent,
  transformer: Transformer,
  autoencoder: Autoencoder,
  diffusion: Diffusion,
  graph: Graph,
  gan: Gan,
  'world-model': WorldModel,
  ssm: Ssm,
  'neural-dynamics': NeuralDynamics,
};

export function FamilyGlyph({ family }: { family: FamilyId }) {
  const G = GLYPHS[family];
  return (
    <svg className="glyph" viewBox="0 0 132 78" width={132} height={78} aria-hidden="true">
      <defs>
        <marker
          id="g-arrow"
          viewBox="0 0 6 6"
          refX="5"
          refY="3"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M0 0.5L5 3L0 5.5" fill="none" stroke="currentColor" strokeWidth="1" />
        </marker>
      </defs>
      <G />
    </svg>
  );
}
