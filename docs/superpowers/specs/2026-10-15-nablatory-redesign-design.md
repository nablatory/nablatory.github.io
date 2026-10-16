# Nablatory redesign: visual system, Guide and Lab

Date: 2026-10-15. Status: implemented in this repository; `docs/DESIGN.md` is
the maintained summary.

## Why

Screenshots of every page at 1440, 900 and 390 px (light and dark) showed one
interface for two audiences. A first-time visitor landed on a form, a graph and
an empty inspector, all competing at the same weight. The model page spent
240 px on a title block before any control, and every tab was equally dense.
Charts had fixed pixel widths, fractional tick labels on an integer step axis
and duplicate tick labels after rounding. Procedure controls sat in cards
separate from the results they produce, so "Not computed yet." never said what
to do. Three typefaces, five label styles and twenty ⓘ marks per page carried
no information.

## Who it is for, and what it has to do

Students meeting an architecture for the first time, and practitioners or
researchers who want to see what a model actually computes. The page's job is
the same for both: open a model, watch it compute and learn, and let the
reader inspect any number. The two audiences differ in how much they want
shown at once, not in what is computed. Both levels therefore run on the same
session in the same Web Worker; switching never resets training.

## Direction: an instrument, not a brochure

The subject is computation you can see: tensors as grids of cells, gradients
as colour, networks as blocks. The interface should read like a well-made
measurement instrument: quiet neutral surfaces, data in colour, chrome in
grey, and one accent reserved for "you can act on this".

### Colour

| Token | Light | Dark | Role |
|---|---|---|---|
| `--paper` | `#F4F5F7` | `#0E1217` | page |
| `--surface` | `#FFFFFF` | `#161B22` | cards, panels, controls |
| `--well` | `#F8F9FB` | `#1C222B` | inset areas, plot backgrounds |
| `--line` | `#E3E6EA` | `#2A323D` | hairlines |
| `--line-strong` | `#C7CDD4` | `#3B4654` | control borders, axes |
| `--ink` | `#141A22` | `#E7EBF0` | text |
| `--ink-2` | `#4B5563` | `#B5BEC9` | secondary text |
| `--ink-3` | `#66717D` | `#8793A0` | captions, ticks (≥ 4.5:1 on paper) |
| `--accent` | `#0F766E` | `#3DBDB2` | primary buttons, selection, active tab, links |
| `--neg` / `--pos` | `#1F4E8C` / `#B0263C` | `#6E9BE0` / `#E06A7E` | diverging data map, class 0 / 1 |
| `--grad` | `#B5740A` | `#E2A43A` | gradients and the backward pass |
| `--cmp` | `#6D4DC2` | `#A48BEF` | side B in comparisons |
| `--ok` / `--bad` | `#1E7F4F` / `#B42334` | `#4CC38A` / `#F0788A` | pass / fail |

Data colours keep their meaning from the previous design (diverging
`--neg → --surface → --pos`, sequential `--surface → --ink`, greyscale for
images). The accent is teal so it can never be mistaken for a data colour.
Side B moves from viridian to violet for the same reason.

### Type

One family and its monospace sibling, so a number in a label and the same
number in a tensor grid share a skeleton.

- IBM Plex Sans (variable): interface, body, headings. Weights 400 / 500 / 600.
- IBM Plex Mono 400 / 500: tensor grids, shapes, operation names, keys.
- KaTeX for equations.

Scale (px / line): 12/16 caption · 13/18 interface · 14/21 body · 15/22 lede ·
16/22 card title (600) · 20/26 section (600) · 26/32 page title (600) ·
36/40 display (600, −0.02 em). Numbers use tabular figures everywhere.
Sentence case; no all-caps labels; no eyebrow labels.

### Space and shape

4 px unit. Page gutter 24 px (16 px under 720 px). Cards and panels: 1 px
`--line`, 8 px radius, 16 px padding, no shadow. Controls: 32 px tall (28 px
compact), 6 px radius, 1 px `--line-strong`; the primary button is filled with
the accent. Focus: 2 px accent ring, 2 px offset. Popovers only carry a shadow.
The architecture canvas keeps its faint engineering grid; it is the one
decorative element and it carries scale.

## Pages

### Top bar (56 px)

Wordmark `∇ Nablatory` · nav: Models, Compare, How it works · right: engine
status pill and a theme button (sun / moon / system icon, labelled by title).

### Home `/`

1. Hero, two columns at ≥ 1100 px: headline "Neural networks you can open
   up.", a two-sentence lede, and two buttons: **Start with the guide**
   (`/m/mlp`) and **Open the lab** (`/m/mlp/lab`). Beside it, the live
   multilayer perceptron on two spirals: plot, step / loss / accuracy, loss
   curve, Train · Step · Reset. The demo is the most characteristic thing on
   the site and stays the first thing seen.
2. One row of three plain statements (no cards): computed on your device;
   every number is computed, nothing is drawn by hand; gradients are checked
   against finite differences.
3. Catalog: a sticky family index on the left (≥ 1100 px), family sections on
   the right. Each family keeps its glyph, name and one-line summary; models
   are cards (name, authors and year, "Published size" or "Reduced scale",
   "Trained weights" when a checkpoint ships).

### Model page: shared header

Row 1: "Models" back link · model name (26/600) · family · fidelity tag ·
"Trained weights" tag when a checkpoint is loaded · right: **Guide | Lab**
switch.

Row 2, sticky: input picker (and Draw where the model accepts a drawing) ·
**Step** · **Train** [N] steps / **Pause** with progress · Reset · readouts:
step, loss, first metric · busy indicator. The same bar serves both levels.

The equation moves to the Notes tab (Lab) and the "The equation" disclosure
(Guide). The one-line summary is shown in the Guide only.

### Guide `/m/<id>` (basic level)

One column, max 960 px, in reading order:

1. **What it does**: two or three plain sentences (`guide.what`), with the
   defining equation behind a disclosure.
2. **Watch it learn**: the model's main views (`guide.views`, default the
   first two) rendered large, beside a loss card (step, loss, metric, loss
   curve). Under it, `guide.watch` says what to look at while training.
3. **The network**: the module graph as a strip of named blocks with shapes
   (input → … → loss), with one sentence per block where the model's `math`
   provides one. Clicking a block opens the Lab with that block selected.
4. **Try changing**: up to three suggested experiments (`guide.try`), each a
   label, one sentence on what to expect, and **Apply**, which sets the
   configuration and optionally trains N steps. "Restore defaults" resets.
5. **Go further**: a link to the Lab, and the model's limitations.

`ModelDef.guide` is optional: `{ what, watch, try?, views? }`. When absent,
the Guide uses `summary`, the family summary, the first two views and the
first select or bool field as the suggested change. All 31 models get a
written guide; every sentence must be true of this implementation at this
scale, not of the paper.

### Lab `/m/<id>/lab` (advanced level)

Two panes: a 288 px settings drawer (collapsible; a sheet above the content
under 900 px) and the main area with tabs Network · Outputs · Training ·
Notes.

- Drawer: Model (Architecture / Computation / Data), Training (optimiser,
  batch, clipping, trainable layers), Weights (checkpoints, seed, edits,
  patches), Saved. Accordion rows 40 px with title and current value.
- Network: a readout strip (loss, metrics, parameters, FLOPs, forward and
  backward time) above the canvas; the canvas fills the width; the inspector
  slides in as a 380 px panel when a block or node is selected and closes
  with ×. No empty inspector state.
- Canvas: boxes 160 × 64, 6 px radius; input boxes dashed; loss and output
  boxes tinted; selected boxes carry a 2 px accent ring. Edges that continue
  on the next row end in a small "continues" marker instead of a line routed
  around the boxes; skips across rows are drawn lighter. A legend under the
  canvas names forward, skip and recurrence edges and the continuation mark.
- Outputs: a 12-column card grid. A procedure's arguments and **Run** button
  sit in the header of the first view it feeds; its other views offer the
  same **Run** in their empty state, so an empty card always says what to
  do. Card spans follow the view kind (images 4, bars 4, charts 6, feature
  maps 6 or 12, strips 12).
- Training: stacked sections rather than inner tabs: loss (full width,
  responsive), metrics, gradient flow, and the last step in detail.
- Notes: equations, scale, limitations, training, data, trained weights,
  references, in a 70 ch column.

### Compare `/compare`

Same components. Two configuration cards with A (ink) and B (violet) bars, a
toolbar, and result tabs whose charts fill the width.

### How it works `/about`

Same page structure; the gradient-check table gains pass / fail pills.

## Charts and cards

- `LineChart` fills its container (ResizeObserver) unless given a width.
  Integer step axes get integer ticks; tick labels are formatted with the
  precision of the tick step so neighbouring labels never repeat; axis names
  sit clear of the tick labels; the legend reads "training loss 0.671 at
  step 12".
- Card header: title (16/600), a small "?" that toggles an explanation line
  under the title (keyboard and touch friendly, unlike hover tooltips), and
  right-aligned actions.
- Empty states name the action: "Run ‹procedure› to see this." with the
  button, or the view's `unavailable` text.

## Narrow screens

Under 720 px the header stacks, the action bar wraps to two rows, tabs scroll
horizontally, cards are single column, the canvas becomes the existing
vertical list, and charts fill the width. Nothing scrolls horizontally at the
page level.

## Accessibility floor

Visible 2 px focus rings; text contrast ≥ 4.5:1; tabs, radios and tooltips
keep their ARIA roles; canvas boxes stay keyboard operable; reduced motion
removes transitions; the skip link stays.

## Technical changes made with the redesign

- Worker commands are serialised per session, so a configuration change can
  no longer interleave with a running training loop.
- Chart tick formatting derives its precision from the tick step.
- `ParamStore.set` checks shape, not only size.
- `stats()` uses a two-pass variance.
- Switching the optimiser kind sets that optimiser's own default smoothing
  constant (0.99 for RMSProp, 0.999 for Adam and AdamW).

## Out of scope

Engine semantics, model computations, the worker protocol, persistence
formats and the test suite are unchanged except where a review finding
required a fix, which the final report lists.

## Changes made while implementing

Screenshots of the first implementation led to these departures from the text
above; `docs/DESIGN.md` describes the result.

- The back link reads "All models", and the Guide column is 1040 px wide so
  that two views fit side by side.
- The Guide shows the defining equation beside "What it does" instead of
  behind a disclosure, and the loss card spans the full width under the views.
- "The network" in the Guide is the same computation graph as the Lab's,
  without the operation drill-down. Selecting a block shows its equation and
  size under the graph, with a button that opens it in the Lab, instead of
  leaving the Guide at once.
- "Try changing" restarts from the page's opening state (default settings,
  the default checkpoint or seed 1) before applying the change and training.
  The suggested changes were measured from that state, so the result on
  screen is the one the note describes. Pause interrupts the run.
- Procedures behind the Guide's views run automatically once. Procedures
  faster than 300 ms re-run after each change and between training chunks;
  slower ones keep their result and say at which step it was computed.
  Procedures the Guide text names in quotes get their own cards.
- The Lab's Training tab uses the card grid (loss and metrics, gradient norm
  and gradient flow, the last step) instead of stacked sections.
- Lab pages are emitted as static routes (`/m/<id>/lab/`) so that direct
  loads return HTTP 200 on GitHub Pages.

## Second pass: graph, Lab organisation and term explanations

Feedback on the first implementation: the network drawing was hard to read,
the Lab still felt crowded, and terms needed short explanations in place.

- The row-wrapped architecture canvas is replaced by a layered top-to-bottom
  graph with live output tiles, joins for sums and concatenations, skip and
  recurrence lanes, a gradients mode and hover lineage (`docs/DESIGN.md`,
  "The computation graph"). Block grouping changed where the old canvas
  showed noise: unscoped reshapes and concatenations join the block that
  uses them, tensors attached only for display are left out, the block that
  holds the objective shows the objective, and a block's output is the
  tensor it passes on. Latent ODE scopes its target, trajectory and loss
  terms, which brings its graph from 58 blocks to 10.
- The Lab's drawer is grouped by what a change does, each group with a
  one-line description. The Network tab keeps the graph and a details panel
  (overview or inspector) side by side; Outputs and Training are split into
  titled sections; the gradient check has its own tab.
- Terms explain themselves on hover, focus or tap from one glossary of
  about 140 entries.
- Long vectors in the inspector are drawn on several lines (32 values per
  line, or a divisor of the length near 32) instead of a one-pixel strip.
