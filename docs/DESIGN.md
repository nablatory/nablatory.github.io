# Nablatory — visual system

The interface is an instrument for reading computation: tensors as grids of
cells, gradients as colour, networks as blocks. It is quiet and exact, with
colour reserved for meaning, so that the numbers and plots are the loudest
thing on every page. The design spec that introduced this system is
`docs/superpowers/specs/2026-10-15-nablatory-redesign-design.md`; this file
is the maintained summary.

## Colour

Tokens live in `src/ui/tokens.css`, with a light and a dark set. Neutral
surfaces carry no hue; one teal accent marks the primary action, the current
tab and the current level. Every other colour encodes data.

| Token | Light | Dark | Meaning |
|---|---|---|---|
| `--paper` | `#F4F5F7` | `#0E1217` | page background |
| `--surface` | `#FFFFFF` | `#161B22` | cards, plots; diverging midpoint |
| `--well` | `#F8F9FB` | `#1C222B` | recessed fields, empty states |
| `--line` | `#E3E6EA` | `#2A323D` | hairlines between regions |
| `--ink` | `#141A22` | `#E7EBF0` | text, forward data flow |
| `--accent` | `#0F766E` | `#3DBDB2` | primary action, current tab and level |
| `--neg` | `#1F4E8C` | `#6E9BE0` | negative values |
| `--pos` | `#B0263C` | `#E06A7E` | positive values |
| `--grad` | `#B5740A` | `#E2A43A` | gradients, backward pass |
| `--cmp` | `#6D4DC2` | `#A48BEF` | configuration B in comparisons |
| `--edge` | `#8C96A2` | `#5F6B78` | lines of the computation graph |
| `--ok`, `--bad` | `#1E7F4F`, `#B42334` | `#4CC38A`, `#F0788A` | check results |

Signed tensors (values and gradients) use the diverging map
`--neg → --surface → --pos`, symmetric around zero, with the scale printed
beside every plot. Whether a heatmap shows values or gradients is stated by
its title, never by colour alone. Non-negative quantities (probabilities,
attention weights) use one ramp from `--surface` to `--ink`. Raw images use
greyscale and say so. A loss curve with one training phase is drawn in grey
with validation points in ink; colour is used only to tell phases apart.

Measured values (timings) carry a filled dot ●; analytic values (FLOP and
parameter counts) carry a hollow dot ○.

## Type

* **IBM Plex Sans** (variable weight) for everything: headings, interface,
  prose and readouts, with tabular figures.
* **IBM Plex Mono** for tensor entries, shapes, axis ticks and configuration
  values, where columns of digits must align.
* **KaTeX** for equations.

Scale (px): 12 caption · 13 interface · 14 body · 15 lede · 16 card title ·
20 section · 26 page title · 36 display. Prose is limited to about 70
characters per line. Sentence case everywhere; no all-caps labels.

## Space and shape

4 px base unit: 4, 8, 12, 16, 24, 32, 48, 64. Cards have 8 px radii and a
hairline border, controls 6 px, chips 4 px. No drop shadows on resting
surfaces; floating layers (tips, menus) have one soft shadow. The
computation graph sits on a faint 16 px dot grid, the one decorative element,
and it carries scale.

Content is laid out on a 12-column card grid. A card's span follows what it
shows: images and bars 4, charts 6, wide feature maps and strips 12. Below
900 px every card spans the full width.

## Motion

120 ms ease-out for selection and disclosure, 200 ms for panel changes.
Numbers never animate between values: a recomputed value replaces the old one
and the difference is shown explicitly. `prefers-reduced-motion` removes all
transitions.

## Pages

```
┌ ∇ Nablatory   Models  Compare  How it works          On your device  ◐ ┐
├────────────────────────────────────────────────────────────────────────┤
│ ‹ All models                                                           │
│ Model name                                         ┌ Guide ┬ Lab ────┐ │
│ Family · fidelity · trained weights                └───────┴─────────┘ │
├────────────────────────────────────────────────────────────────────────┤
│ [Settings]  Input ▾   Train [n] steps  Step  Reset   Step  Loss  metric │ sticky
└────────────────────────────────────────────────────────────────────────┘
```

The header and the action bar are the same at both levels; the bar holds
everything that runs the model. The Settings button exists in the Lab only.

**Guide** (`/m/<id>`), one column of 1040 px in reading order:

1. *What it does*: the model in plain words, its source, and the defining
   equation beside it.
2. *Watch it learn*: what to look for while training, the model's main views
   and the loss curve. Procedures behind these views run once when missing;
   fast ones (under 300 ms) re-run after every change and between training
   chunks, slow ones say "Computed at step N, before the latest change."
   Procedures the text names in quotes get their own cards.
3. *The network*: the computation graph; selecting a block shows its
   equation and size, with a button that opens it in the Lab.
4. *Try changing*: up to three changes. Each restarts from the page's opening
   state (default settings and starting weights), applies the change and,
   where stated, trains, so the result matches the run the text describes.
5. *Go further*: the Lab, and what this version leaves out.

**Lab** (`/m/<id>/lab`): a 288 px settings drawer (open by default from
1100 px, above the content below 900 px) beside five tabs.

* The drawer groups settings by what a change does, each group with one line
  saying so: *Architecture* (rebuilds the network from random weights),
  *Computation* (switches in the forward pass and the loss; the weights stay),
  *Data*, *Optimiser*, *Weights* (where the parameters came from) and
  *Saved*.
* *Network*: the computation graph, with a details panel beside it (below it
  under 1100 px). The panel shows an overview of the last forward pass (loss,
  metrics, sizes, timings and the blocks that changed most) until a block or
  tensor is selected; then it shows the inspector, with a way back.
* *Outputs*: one section for the views recomputed after every change, then
  one section per procedure, with its description, arguments and Run button
  in the section header and its views below.
* *Training*: sections *Loss and metrics*, *Gradients* (norm and flow per
  parameter group) and *Last step in detail*.
* *Gradient check* and *Notes* (equations, fidelity, limitations).

**Home** opens with the live multilayer perceptron, then a sticky family index
beside the catalog. **Compare** and **How it works** use the same cards,
tables and pills.

Below 720 px the action bar wraps and stops being sticky, tips dock to the
bottom of the screen, and nothing scrolls horizontally at the page level; a
graph too wide for its frame scrolls inside it.

## The computation graph

`src/viz/NetworkGraph.tsx` draws the module graph of the last forward pass
(`src/viz/architecture.ts`) top to bottom with a layered layout
(`src/viz/graphLayout.ts`): longest-path layers, sources placed just above
their first use, barycentre ordering then neighbour swaps against crossings,
and positions from isotonic regression so that blocks sit between their
inputs and outputs. Long edges run through the gaps between blocks and are
straightened wherever one x fits; an input reused three or more layers down
is drawn again as a small dashed reference next to that use.

* A block is a card: a tile of its output for the first example (four
  channels at most, on its own scale), its name, its main operations, and
  its output shape and parameter count. Inputs are dashed, parameter-free
  steps sit on the recessed fill, additions, concatenations and products
  with several inputs are small ⊕, ‖ and ⊗ joins, and the loss is a card
  with ℒ and its value.
* Line width is the number of values passed (log scale). State carried to
  the next time step runs in dashed lanes on the right; a value passed on
  with its gradient stopped is dotted; a block that adds its input back
  (x + F(x) inside one block) carries a short arc on its left side.
* *Gradients* mode shows ∂L/∂y on the tiles, the gradient norm on each block,
  line widths from the norm of the gradient carried, and arrows pointing
  backwards; blocks off the gradient path say "no gradient".
* Hover lights a block's ancestors and descendants; + lists a block's
  operations, each selectable; a teal bar marks blocks whose output changed.
* Blocks are 224 px wide when the graph fits, 168 px otherwise; a graph still
  too wide is scaled down to 78 % and then scrolls inside its frame.

## Term explanations

Terms with a dotted underline explain themselves on hover, keyboard focus or
tap: a name, what the thing does, and why it is there, in a sentence or two
each (`src/ui/glossary.ts`). Settings, metrics, card titles, graph blocks,
operations and the first mentions of common terms in guide text carry them.
Equations stay in the inspector; the explanations are meant to be read at a
glance.

## Words

Labels name what a control does, in sentence case. A card's explanation sits
behind its "?" button, never as a paragraph above a control. Empty states say
what to press. When something cannot be computed, the view says so instead of
showing a placeholder. No marketing language.
