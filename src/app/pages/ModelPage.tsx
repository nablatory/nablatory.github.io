import { useEffect, useMemo } from 'react';
import type { ModelDef } from '../../models/types';
import { Experiment } from '../experiment/controller';
import { useStore } from '../../state/store';
import { Button } from '../../ui/controls';
import type { ModelView } from '../router';
import { ModelHeader } from './model/shared';
import { GuidePage } from './model/Guide';
import { LabPage } from './model/Lab';

/**
 * One model at two levels: the Guide (plain language, a few controls) and the
 * Lab (every tensor and setting). Both share one experiment, so switching
 * level keeps the weights, the settings and the training history.
 */
export function ModelPage({ model, view }: { model: ModelDef; view: ModelView }) {
  const exp = useMemo(() => new Experiment(model.id), [model.id]);
  useEffect(() => {
    exp.open();
    return () => exp.close();
  }, [exp]);
  useEffect(() => {
    document.title = `${model.name}${view === 'lab' ? ', lab' : ''} · Nablatory`;
  }, [model.name, view]);

  const status = useStore(exp.store, (s) => s.status);
  const error = useStore(exp.store, (s) => s.error);

  return (
    <main className={`model-page model-page-${view}`} id="main">
      <ModelHeader model={model} view={view} />
      {status === 'error' ? (
        <div className="ws-error" role="alert">
          <p>The model could not be opened: {error}</p>
          <Button onClick={() => exp.open()}>Try again</Button>
        </div>
      ) : view === 'guide' ? (
        <GuidePage exp={exp} model={model} />
      ) : (
        <LabPage exp={exp} model={model} />
      )}
    </main>
  );
}
