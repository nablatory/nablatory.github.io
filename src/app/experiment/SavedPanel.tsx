import { useEffect, useState } from 'react';
import type { ModelDef } from '../../models/types';
import type { Experiment } from './controller';
import {
  deleteExperiment,
  exportJson,
  importJson,
  listExperiments,
  saveExperiment,
  type SavedExperiment,
} from '../../persist/experiments';
import { Button, Toggle } from '../../ui/controls';
import { RailSection } from './Rail';
import { useStore } from '../../state/store';

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Experiments stored in this browser (IndexedDB); export/import as versioned JSON. */
export function SavedPanel({ exp, model }: { exp: Experiment; model: ModelDef }) {
  const summary = useStore(exp.store, (s) => s.summary);
  const busy = useStore(exp.store, (s) => s.busy);
  const [items, setItems] = useState<SavedExperiment[]>([]);
  const [title, setTitle] = useState('');
  const [withWeights, setWithWeights] = useState(true);
  const [note, setNote] = useState<string | null>(null);
  const refresh = () =>
    listExperiments(model.id)
      .then(setItems)
      .catch(() => setNote('Browser storage is unavailable; use export instead.'));
  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model.id]);

  const snapshot = async () => {
    const e = await exp.snapshot(
      title.trim() || `${model.name}, step ${summary?.step ?? 0}`,
      withWeights,
    );
    return e;
  };

  return (
    <RailSection title="Saved" meta={items.length ? String(items.length) : undefined}>
      <p className="rail-note">Kept in this browser only.</p>
      <input
        className="text-input"
        placeholder="Title"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        aria-label="Experiment title"
      />
      <Toggle checked={withWeights} onChange={setWithWeights} label="Include weights" />
      <div className="row">
        <Button
          size="sm"
          disabled={!summary || !!busy}
          onClick={async () => {
            const e = await snapshot();
            if (!e) return;
            await saveExperiment(e).catch(() => setNote('Could not write to browser storage.'));
            setTitle('');
            refresh();
          }}
        >
          Save
        </Button>
        <Button
          size="sm"
          variant="quiet"
          disabled={!summary || !!busy}
          onClick={async () => {
            const e = await snapshot();
            if (e) download(`${e.model}-${e.created.slice(0, 10)}.nablatory.json`, exportJson(e));
          }}
        >
          Export
        </Button>
        <label className="btn btn-quiet btn-sm file-btn">
          Import…
          <input
            type="file"
            accept="application/json,.json"
            onChange={async (ev) => {
              const f = ev.target.files?.[0];
              if (!f) return;
              try {
                const e = importJson(await f.text());
                if (e.model !== model.id) throw new Error(`This file is for model “${e.model}”.`);
                await saveExperiment(e).catch(() => undefined);
                await exp.restore(e);
                refresh();
              } catch (err) {
                setNote(err instanceof Error ? err.message : String(err));
              }
            }}
          />
        </label>
      </div>
      {note && <p className="field-hint is-bad">{note}</p>}
      <ul className="saved-list">
        {items.map((e) => (
          <li key={e.id}>
            <button
              type="button"
              className="linkish"
              onClick={() => exp.restore(e)}
              disabled={!!busy}
            >
              {e.title}
            </button>
            <span className="muted">
              {new Date(e.created).toLocaleDateString()}, step {e.step}
            </span>
            <span className="row">
              <Button
                size="sm"
                variant="quiet"
                onClick={() =>
                  download(`${e.model}-${e.id.slice(0, 8)}.nablatory.json`, exportJson(e))
                }
              >
                Export
              </Button>
              <Button
                size="sm"
                variant="quiet"
                onClick={async () => {
                  await deleteExperiment(e.id);
                  refresh();
                }}
              >
                Delete
              </Button>
            </span>
          </li>
        ))}
      </ul>
    </RailSection>
  );
}
