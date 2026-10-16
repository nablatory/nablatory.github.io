import type { ModelDef } from '../types';
import neuralOde from './neural-ode';
import latentOde from './latent-ode';

export const models: ModelDef[] = [neuralOde, latentOde];
