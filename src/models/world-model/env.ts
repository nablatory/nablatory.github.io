import { Rng, Tensor } from '../../engine';
import type { Batch, Config, Dataset } from '../types';

/**
 * Point-mass reaching task, simulated exactly in TypeScript.
 *
 * State s = (x, y, vx, vy) in the arena [−1, 1]², action a ∈ [−1, 1]²
 * (a force), time step Δt = 0.1. Semi-implicit Euler with linear drag:
 *   v_{t+1} = v_t + Δt (k a_t − c v_t),      k = 2, c = 1
 *   p_{t+1} = p_t + Δt v_{t+1}
 * Walls: a coordinate that leaves [−1, 1] is clamped and its velocity set to 0.
 * Reward of a state: r(s) = exp(−‖p − g‖² / (2 w²)), goal g = (0.5, 0.5), w = 0.3.
 * Observation: a 16 × 16 greyscale image; the mass is a Gaussian blob
 * (σ = 0.2, peak 1) and the goal a fixed dim dot (σ = 0.08, peak 0.35);
 * pixel value = max of the two. Velocity is not visible in one frame.
 */
export const ENV = {
  dt: 0.1,
  gain: 2,
  drag: 1,
  goal: [0.5, 0.5] as [number, number],
  rewardWidth: 0.3,
  size: 16,
  blob: 0.2,
  goalBlob: 0.08,
  goalPeak: 0.35,
  stateDim: 4,
  actionDim: 2,
  episodeLength: 40,
};

export const OBS_DIM = ENV.size * ENV.size;

export function clampAction(a: number): number {
  return Math.max(-1, Math.min(1, a));
}

export function envStep(s: ArrayLike<number>, a: ArrayLike<number>): number[] {
  const { dt, gain, drag } = ENV;
  const next = [0, 0, 0, 0];
  for (let d = 0; d < 2; d++) {
    let v = s[2 + d] + dt * (gain * clampAction(a[d]) - drag * s[2 + d]);
    let p = s[d] + dt * v;
    if (p > 1) {
      p = 1;
      v = 0;
    }
    if (p < -1) {
      p = -1;
      v = 0;
    }
    next[d] = p;
    next[2 + d] = v;
  }
  return next;
}

export function envReward(s: ArrayLike<number>): number {
  const dx = s[0] - ENV.goal[0];
  const dy = s[1] - ENV.goal[1];
  return Math.exp(-(dx * dx + dy * dy) / (2 * ENV.rewardWidth * ENV.rewardWidth));
}

/** Render a state into `out[off .. off + 256)`, row 0 at the top (y = +1). */
export function render(s: ArrayLike<number>, out: Float32Array | Float64Array, off = 0): void {
  const n = ENV.size;
  const b2 = 2 * ENV.blob * ENV.blob;
  const g2 = 2 * ENV.goalBlob * ENV.goalBlob;
  for (let i = 0; i < n; i++) {
    const py = 1 - ((i + 0.5) / n) * 2;
    for (let j = 0; j < n; j++) {
      const px = -1 + ((j + 0.5) / n) * 2;
      const m = Math.exp(-((px - s[0]) ** 2 + (py - s[1]) ** 2) / b2);
      const g = ENV.goalPeak * Math.exp(-((px - ENV.goal[0]) ** 2 + (py - ENV.goal[1]) ** 2) / g2);
      out[off + i * n + j] = Math.max(m, g);
    }
  }
}

export interface Episode {
  /** [L + 1, 4] */
  states: Float32Array;
  /** [L, 2] */
  actions: Float32Array;
  /** [L + 1]: r(s_t) */
  rewards: Float32Array;
  /** [L + 1, 256] */
  obs: Float32Array;
  length: number;
}

/**
 * Scripted behaviour policy that produced the fixed dataset: a PD controller
 * towards a waypoint (the goal with probability ½, otherwise a uniform random
 * point; re-drawn every 10 steps) plus Ornstein–Uhlenbeck noise,
 *   a_t = clip(1.5 (w − p_t) − 0.8 v_t + n_t),  n_t = 0.8 n_{t−1} + 0.4 ε_t.
 */
export function generateEpisode(rng: Rng, L = ENV.episodeLength): Episode {
  const states = new Float32Array((L + 1) * 4);
  const actions = new Float32Array(L * 2);
  const rewards = new Float32Array(L + 1);
  const obs = new Float32Array((L + 1) * OBS_DIM);
  let s = [rng.range(-0.9, 0.9), rng.range(-0.9, 0.9), 0, 0];
  let w = [0, 0];
  const noise = [0, 0];
  for (let t = 0; t <= L; t++) {
    states.set(s, t * 4);
    rewards[t] = envReward(s);
    render(s, obs, t * OBS_DIM);
    if (t === L) break;
    if (t % 10 === 0) {
      w =
        rng.uniform() < 0.5
          ? [ENV.goal[0] + 0.1 * rng.normal(), ENV.goal[1] + 0.1 * rng.normal()]
          : [rng.range(-0.9, 0.9), rng.range(-0.9, 0.9)];
    }
    const a = [0, 0];
    for (let d = 0; d < 2; d++) {
      noise[d] = 0.8 * noise[d] + 0.4 * rng.normal();
      a[d] = clampAction(1.5 * (w[d] - s[d]) - 0.8 * s[2 + d] + noise[d]);
    }
    actions.set(a, t * 2);
    s = envStep(s, a);
  }
  return { states, actions, rewards, obs, length: L };
}

export const DATASET = { trainEpisodes: 96, valEpisodes: 8, trainSeed: 501, valSeed: 502 };

let cache: { train: Episode[]; val: Episode[] } | null = null;

export function episodes(): { train: Episode[]; val: Episode[] } {
  if (!cache) {
    const tr = new Rng(DATASET.trainSeed);
    const va = new Rng(DATASET.valSeed);
    cache = {
      train: Array.from({ length: DATASET.trainEpisodes }, () => generateEpisode(tr)),
      val: Array.from({ length: DATASET.valEpisodes }, () => generateEpisode(va)),
    };
  }
  return cache;
}

/**
 * Batch of B subsequences of length T. Index t of each sequence holds the
 * observation o_t, the state s_t, the reward r_t = r(s_t) and the action a_t
 * taken at s_t (which leads to s_{t+1}).
 *   obs [B, T, 256], state [B, T, 4], action [B, T, 2], reward [B, T]
 */
export function sequenceBatch(
  eps: Episode[],
  picks: { ep: number; start: number }[],
  T: number,
  caption: string,
): Batch {
  const B = picks.length;
  const obs = new Float32Array(B * T * OBS_DIM);
  const state = new Float32Array(B * T * 4);
  const action = new Float32Array(B * T * 2);
  const reward = new Float32Array(B * T);
  picks.forEach(({ ep, start }, b) => {
    const e = eps[ep];
    for (let t = 0; t < T; t++) {
      const k = start + t;
      obs.set(e.obs.subarray(k * OBS_DIM, (k + 1) * OBS_DIM), (b * T + t) * OBS_DIM);
      state.set(e.states.subarray(k * 4, k * 4 + 4), (b * T + t) * 4);
      action.set(e.actions.subarray(k * 2, k * 2 + 2), (b * T + t) * 2);
      reward[b * T + t] = e.rewards[k];
    }
  });
  return {
    inputs: {
      obs: new Tensor([B, T, OBS_DIM], obs),
      state: new Tensor([B, T, 4], state),
      action: new Tensor([B, T, 2], action),
      reward: new Tensor([B, T], reward),
    },
    meta: { caption },
  };
}

export interface SequenceConfig extends Config {
  seqLen: number;
}

export function pointMassDataset<C extends Config>(seqLen: (c: C) => number): Dataset<C> {
  const L = ENV.episodeLength;
  return {
    id: 'point-mass',
    name: 'Point-mass reaching trajectories',
    description: `${DATASET.trainEpisodes} training and ${DATASET.valEpisodes} held-out episodes of ${L} steps, simulated in the browser from a seeded scripted policy (PD control towards random waypoints or the goal, plus Ornstein–Uhlenbeck noise). Observations are 16 × 16 renderings; rewards measure closeness to a fixed goal.`,
    source: 'synthetic',
    train(rng, batchSize, c) {
      const T = seqLen(c);
      const { train } = episodes();
      const picks = Array.from({ length: batchSize }, () => ({
        ep: rng.int(train.length),
        start: rng.int(L - T + 1),
      }));
      return sequenceBatch(train, picks, T, `${batchSize} training subsequences of ${T} steps`);
    },
    validation(c) {
      const T = seqLen(c);
      const { val } = episodes();
      return sequenceBatch(
        val,
        val.map((_, i) => ({ ep: i, start: (i * 7) % (L - T + 1) })),
        T,
        'held-out subsequences',
      );
    },
    exampleCount: DATASET.valEpisodes,
    example(index, c) {
      const T = seqLen(c);
      return sequenceBatch(
        episodes().val,
        [{ ep: index, start: 0 }],
        T,
        `held-out episode ${index + 1}, steps 0–${T - 1}`,
      );
    },
    exampleLabel(index) {
      return `Held-out episode ${index + 1}`;
    },
  };
}

/** Images of states [T, 1, 16, 16]. */
export function renderStates(states: number[][]): Tensor {
  const out = new Float32Array(states.length * OBS_DIM);
  states.forEach((s, t) => render(s, out, t * OBS_DIM));
  return new Tensor([states.length, 1, ENV.size, ENV.size], out);
}

/** Scatter points [path…, goal] with colour classes 0 (path), 1 (start), 2 (goal). */
export function pathScatter(states: number[][]): { points: Tensor; colors: Tensor } {
  const n = states.length + 1;
  const p = new Float32Array(n * 2);
  const c = new Float32Array(n);
  states.forEach((s, i) => {
    p[i * 2] = s[0];
    p[i * 2 + 1] = s[1];
    c[i] = i === 0 ? 1 : 0;
  });
  p[(n - 1) * 2] = ENV.goal[0];
  p[(n - 1) * 2 + 1] = ENV.goal[1];
  c[n - 1] = 2;
  return { points: new Tensor([n, 2], p), colors: new Tensor([n], c) };
}

export const ENV_EXTENT: [number, number, number, number] = [-1.05, 1.05, -1.05, 1.05];

export const ENV_MATH = {
  tex: 'v_{t+1} = v_t + \\Delta t\\,(k a_t - c\\, v_t),\\quad p_{t+1} = \\operatorname{clip}(p_t + \\Delta t\\, v_{t+1}),\\quad r_t = e^{-\\lVert p_t - g \\rVert^2 / 2w^2}',
  text: 'Point mass in [−1, 1]² with drag; Δt = 0.1, k = 2, c = 1, goal g = (0.5, 0.5), w = 0.3. Observations are 16 × 16 images of the mass and the goal.',
};
