/**
 * Layered (Sugiyama-style) layout of a directed graph, top to bottom.
 *
 * 1. Layers: longest path from the sources, then every node with successors is moved down to just
 *    above its nearest successor, so inputs and side branches sit next to where they are used.
 * 2. Edges spanning several layers pass through dummy nodes, one per layer crossed.
 * 3. Order within layers: barycentre sweeps, keeping the order with the fewest crossings.
 * 4. Horizontal positions: alternating sweeps that pull each node towards the mean of its
 *    neighbours, solved per layer as an isotonic regression with minimum separations.
 * Feedback edges (to an earlier node) are kept out of steps 1–4 and routed in lanes on the right.
 */

export interface LayoutNode {
  id: string;
  w: number;
  h: number;
  /** Horizontal position of the edge ports, from the left of the box (default w / 2). */
  anchor?: number;
}

export interface LayoutEdge {
  from: string;
  to: string;
  feedback?: boolean;
}

export interface Point {
  x: number;
  y: number;
}

export interface PlacedNode extends LayoutNode {
  x: number;
  y: number;
  layer: number;
  order: number;
}

export interface PlacedEdge {
  from: string;
  to: string;
  feedback: boolean;
  /** Forward: source port, then for each dummy the top and bottom of its layer, then target port.
   *  Feedback: source's right side, two lane corners, target's right side. */
  points: Point[];
}

export interface Layout {
  nodes: Map<string, PlacedNode>;
  edges: PlacedEdge[];
  width: number;
  height: number;
  layerTop: number[];
  layerHeight: number[];
}

export interface LayoutOptions {
  layerGap?: number;
  nodeGap?: number;
  dummyGap?: number;
  pad?: number;
  /** Space reserved right of the graph for each feedback lane. */
  laneGap?: number;
  /** Extra width after the last lane (for its label). */
  laneLabel?: number;
}

interface Item {
  id: string;
  dummy: boolean;
  w: number;
  anchor: number;
  layer: number;
  /** Position of the anchor. */
  x: number;
  up: Item[];
  down: Item[];
}

export function layoutGraph(
  nodes: LayoutNode[],
  edges: LayoutEdge[],
  opts: LayoutOptions = {},
): Layout {
  const layerGap = opts.layerGap ?? 44;
  const nodeGap = opts.nodeGap ?? 28;
  const dummyGap = opts.dummyGap ?? 12;
  const pad = opts.pad ?? 24;
  const laneGap = opts.laneGap ?? 16;
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const fwd = edges.filter(
    (e) => !e.feedback && index.has(e.from) && index.has(e.to) && e.from !== e.to,
  );
  const back = edges.filter((e) => e.feedback && index.has(e.from) && index.has(e.to));

  // Topological order (nodes come in execution order, but do not rely on it).
  const preds = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  const succs = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  for (const e of fwd) {
    preds.get(e.to)!.push(e.from);
    succs.get(e.from)!.push(e.to);
  }
  const indeg = new Map(nodes.map((n) => [n.id, preds.get(n.id)!.length]));
  const queue = nodes.filter((n) => !indeg.get(n.id)).map((n) => n.id);
  const topo: string[] = [];
  while (queue.length) {
    queue.sort((a, b) => index.get(a)! - index.get(b)!);
    const id = queue.shift()!;
    topo.push(id);
    for (const s of succs.get(id)!) {
      indeg.set(s, indeg.get(s)! - 1);
      if (!indeg.get(s)) queue.push(s);
    }
  }
  // A cycle among edges not marked as feedback: place the rest in input order.
  for (const n of nodes) if (!topo.includes(n.id)) topo.push(n.id);

  const layer = new Map<string, number>();
  for (const id of topo)
    layer.set(id, Math.max(0, ...preds.get(id)!.map((p) => (layer.get(p) ?? -1) + 1)));
  // Sources (inputs, labels) move down to just above their first use.
  for (const id of topo) {
    const s = succs.get(id)!;
    if (!preds.get(id)!.length && s.length)
      layer.set(id, Math.max(0, Math.min(...s.map((x) => layer.get(x)!)) - 1));
  }
  const nLayers = Math.max(0, ...layer.values()) + 1;
  // A node with no forward edges at all goes to the least crowded layer (the lower one on a tie).
  const linked = new Set(
    edges.filter((e) => index.has(e.from) && index.has(e.to)).flatMap((e) => [e.from, e.to]),
  );
  for (const id of topo) {
    if (linked.has(id) || nLayers < 2) continue;
    const count = Array.from({ length: nLayers }, () => 0);
    for (const [k, l] of layer) if (k !== id) count[l]++;
    let best = nLayers - 1;
    for (let l = nLayers - 1; l >= 0; l--) if (count[l] < count[best]) best = l;
    layer.set(id, best);
  }

  // Items, with dummies for long edges.
  const items = new Map<string, Item>();
  const layers: Item[][] = Array.from({ length: nLayers }, () => []);
  for (const id of topo) {
    const n = nodes[index.get(id)!];
    const it: Item = {
      id,
      dummy: false,
      w: n.w,
      anchor: n.anchor ?? n.w / 2,
      layer: layer.get(id)!,
      x: 0,
      up: [],
      down: [],
    };
    items.set(id, it);
    layers[it.layer].push(it);
  }
  const chains: { e: LayoutEdge; items: Item[] }[] = [];
  for (const e of fwd) {
    const a = items.get(e.from)!;
    const b = items.get(e.to)!;
    const chain = [a];
    let prev = a;
    for (let l = a.layer + 1; l < b.layer; l++) {
      const d: Item = {
        id: `${e.from}→${e.to}#${l}`,
        dummy: true,
        w: 0,
        anchor: 0,
        layer: l,
        x: 0,
        up: [prev],
        down: [],
      };
      prev.down.push(d);
      layers[l].push(d);
      chain.push(d);
      prev = d;
    }
    prev.down.push(b);
    b.up.push(prev);
    chain.push(b);
    chains.push({ e, items: chain });
  }

  // Ordering: barycentre sweeps.
  const pos = new Map<Item, number>();
  const renumber = () => layers.forEach((L) => L.forEach((it, i) => pos.set(it, i)));
  renumber();
  const crossings = () => {
    let c = 0;
    for (let l = 0; l + 1 < nLayers; l++) {
      const segs: [number, number][] = [];
      for (const it of layers[l]) for (const d of it.down) segs.push([pos.get(it)!, pos.get(d)!]);
      for (let i = 0; i < segs.length; i++)
        for (let j = i + 1; j < segs.length; j++) {
          const [a1, b1] = segs[i];
          const [a2, b2] = segs[j];
          if ((a1 - a2) * (b1 - b2) < 0) c++;
        }
    }
    return c;
  };
  let best = layers.map((L) => [...L]);
  let bestC = crossings();
  for (let iter = 0; iter < 16 && bestC > 0; iter++) {
    const down = iter % 2 === 0;
    const range = down
      ? Array.from({ length: nLayers - 1 }, (_, i) => i + 1)
      : Array.from({ length: nLayers - 1 }, (_, i) => nLayers - 2 - i);
    for (const l of range) {
      const bary = new Map<Item, number>();
      for (const it of layers[l]) {
        const nb = down ? it.up : it.down;
        bary.set(
          it,
          nb.length ? nb.reduce((s, x) => s + pos.get(x)!, 0) / nb.length : pos.get(it)!,
        );
      }
      layers[l].sort((a, b) => bary.get(a)! - bary.get(b)! || pos.get(a)! - pos.get(b)!);
      layers[l].forEach((it, i) => pos.set(it, i));
    }
    const c = crossings();
    if (c < bestC) {
      bestC = c;
      best = layers.map((L) => [...L]);
    }
  }
  best.forEach((L, l) => (layers[l] = L));
  renumber();
  // Then swap neighbours within a layer while that removes crossings.
  for (let pass = 0; pass < 6 && bestC > 0; pass++) {
    let improved = false;
    for (const L of layers)
      for (let i = 0; i + 1 < L.length; i++) {
        const swap = () => {
          [L[i], L[i + 1]] = [L[i + 1], L[i]];
          pos.set(L[i], i);
          pos.set(L[i + 1], i + 1);
        };
        swap();
        const c = crossings();
        if (c < bestC) {
          bestC = c;
          improved = true;
        } else swap();
      }
    if (!improved) break;
  }

  // Horizontal placement.
  const sep = (a: Item, b: Item) =>
    a.w - a.anchor + b.anchor + (a.dummy || b.dummy ? dummyGap : nodeGap);
  for (const L of layers) {
    let x = 0;
    L.forEach((it, i) => {
      if (i > 0) x += sep(L[i - 1], it);
      it.x = x;
    });
  }
  const weight = (a: Item, b: Item) => (a.dummy && b.dummy ? 8 : a.dummy || b.dummy ? 2 : 1);
  const place = (L: Item[], target: (it: Item) => number | null) => {
    if (!L.length) return;
    // Isotonic regression: minimise Σ w (x − t)² subject to x[i+1] ≥ x[i] + sep(i, i+1).
    const off: number[] = [0];
    for (let i = 1; i < L.length; i++) off.push(off[i - 1] + sep(L[i - 1], L[i]));
    const blocks: { sum: number; w: number; n: number }[] = [];
    L.forEach((it, i) => {
      let t = target(it);
      const w = t === null ? 0.05 : 1;
      // A node without neighbours on this side stays packed against the one before it.
      if (t === null)
        t =
          i > 0
            ? L[i - 1].x + sep(L[i - 1], it)
            : i + 1 < L.length
              ? L[i + 1].x - sep(it, L[i + 1])
              : it.x;
      blocks.push({ sum: w * (t - off[i]), w, n: 1 });
      while (blocks.length > 1) {
        const b = blocks[blocks.length - 1];
        const a = blocks[blocks.length - 2];
        if (a.sum / a.w <= b.sum / b.w) break;
        blocks.splice(blocks.length - 2, 2, { sum: a.sum + b.sum, w: a.w + b.w, n: a.n + b.n });
      }
    });
    let i = 0;
    for (const b of blocks) {
      const v = b.sum / b.w;
      for (let k = 0; k < b.n; k++, i++) L[i].x = v + off[i];
    }
  };
  const mean = (it: Item, nb: Item[]) => {
    if (!nb.length) return null;
    let s = 0;
    let w = 0;
    for (const x of nb) {
      const k = weight(it, x);
      s += k * x.x;
      w += k;
    }
    return s / w;
  };
  for (let iter = 0; iter < 10; iter++) {
    for (let l = 1; l < nLayers; l++) place(layers[l], (it) => mean(it, it.up));
    for (let l = nLayers - 2; l >= 0; l--) place(layers[l], (it) => mean(it, it.down));
  }
  // Final pass: each node between its two sides.
  for (let l = 0; l < nLayers; l++)
    place(layers[l], (it) => {
      const a = mean(it, it.up);
      const b = mean(it, it.down);
      return a === null ? b : b === null ? a : (a + b) / 2;
    });
  // Long edges run straight down: each chain of dummies is cut into the fewest runs that fit one
  // x between their neighbours in every layer. The order within layers is kept, so no crossing
  // is added.
  const room = (d: Item): [number, number] => {
    const L = layers[d.layer];
    const i = pos.get(d)!;
    return [
      i > 0 ? L[i - 1].x + sep(L[i - 1], d) : -Infinity,
      i + 1 < L.length ? L[i + 1].x - sep(d, L[i + 1]) : Infinity,
    ];
  };
  const straighten = (run: Item[]) => {
    if (run.length < 2) return;
    let lo = -Infinity;
    let hi = Infinity;
    for (const d of run) {
      const [a, b] = room(d);
      lo = Math.max(lo, a);
      hi = Math.min(hi, b);
    }
    const xs = run.map((d) => d.x).sort((p, q) => p - q);
    const x = Math.min(hi, Math.max(lo, xs[xs.length >> 1]));
    for (const d of run) d.x = x;
  };
  for (const { items: chain } of [...chains].sort((p, q) => q.items.length - p.items.length)) {
    let run: Item[] = [];
    let lo = -Infinity;
    let hi = Infinity;
    for (const d of chain.slice(1, -1)) {
      const [a, b] = room(d);
      if (Math.max(lo, a) > Math.min(hi, b)) {
        straighten(run);
        run = [];
        lo = -Infinity;
        hi = Infinity;
      }
      run.push(d);
      lo = Math.max(lo, a);
      hi = Math.min(hi, b);
    }
    straighten(run);
  }

  let minX = Infinity;
  let maxX = -Infinity;
  for (const it of items.values()) {
    minX = Math.min(minX, it.x - it.anchor);
    maxX = Math.max(maxX, it.x - it.anchor + it.w);
  }
  for (const L of layers)
    for (const it of L) {
      minX = Math.min(minX, it.x);
      maxX = Math.max(maxX, it.x);
    }
  const shift = pad - minX;

  // Vertical placement.
  const layerHeight = layers.map((L) =>
    Math.max(0, ...L.filter((it) => !it.dummy).map((it) => nodes[index.get(it.id)!].h)),
  );
  const layerTop: number[] = [];
  let y = pad;
  for (let l = 0; l < nLayers; l++) {
    layerTop.push(y);
    y += (layerHeight[l] || 8) + layerGap;
  }
  const height = y - layerGap + pad;

  const placed = new Map<string, PlacedNode>();
  for (const it of items.values()) {
    const n = nodes[index.get(it.id)!];
    placed.set(it.id, {
      ...n,
      x: it.x + shift - it.anchor,
      y: layerTop[it.layer] + (layerHeight[it.layer] - n.h) / 2,
      layer: it.layer,
      order: pos.get(it)!,
    });
  }

  // Ports: spread a node's edges along its bottom (outgoing) and top (incoming) side.
  const portX = (p: PlacedNode) => p.x + (p.anchor ?? p.w / 2);
  const outPorts = new Map<string, Map<string, number>>();
  const inPorts = new Map<string, Map<string, number>>();
  const spread = (p: PlacedNode, others: { key: string; x: number }[]) => {
    const m = new Map<string, number>();
    others.sort((a, b) => a.x - b.x);
    const span = Math.min(p.w * 0.6, (others.length - 1) * 14);
    others.forEach((o, i) =>
      m.set(
        o.key,
        portX(p) + (others.length > 1 ? -span / 2 + (span * i) / (others.length - 1) : 0),
      ),
    );
    return m;
  };
  const nextX = (c: Item[], i: number) =>
    c[i].dummy ? c[i].x + shift : portX(placed.get(c[i].id)!);
  for (const id of placed.keys()) {
    const outs = chains.filter((c) => c.e.from === id);
    const ins = chains.filter((c) => c.e.to === id);
    outPorts.set(
      id,
      spread(
        placed.get(id)!,
        outs.map((c) => ({ key: `${c.e.from}→${c.e.to}`, x: nextX(c.items, 1) })),
      ),
    );
    inPorts.set(
      id,
      spread(
        placed.get(id)!,
        ins.map((c) => ({ key: `${c.e.from}→${c.e.to}`, x: nextX(c.items, c.items.length - 2) })),
      ),
    );
  }

  const placedEdges: PlacedEdge[] = chains.map(({ e, items: chain }) => {
    const a = placed.get(e.from)!;
    const b = placed.get(e.to)!;
    const key = `${e.from}→${e.to}`;
    const pts: Point[] = [{ x: outPorts.get(e.from)!.get(key)!, y: a.y + a.h }];
    for (const d of chain.slice(1, -1)) {
      const top = layerTop[d.layer];
      pts.push({ x: d.x + shift, y: top }, { x: d.x + shift, y: top + layerHeight[d.layer] });
    }
    pts.push({ x: inPorts.get(e.to)!.get(key)!, y: b.y });
    return { from: e.from, to: e.to, feedback: false, points: pts };
  });

  let right = maxX + shift;
  back.forEach((e, k) => {
    const a = placed.get(e.from)!;
    const b = placed.get(e.to)!;
    const lane = maxX + shift + laneGap * (k + 1) + 8;
    right = Math.max(right, lane);
    placedEdges.push({
      from: e.from,
      to: e.to,
      feedback: true,
      points: [
        { x: a.x + a.w, y: a.y + a.h / 2 },
        { x: lane, y: a.y + a.h / 2 },
        { x: lane, y: b.y + b.h / 2 },
        { x: b.x + b.w, y: b.y + b.h / 2 },
      ],
    });
  });
  const width = right + pad + (back.length ? (opts.laneLabel ?? 0) : 0);
  return { nodes: placed, edges: placedEdges, width, height, layerTop, layerHeight };
}
