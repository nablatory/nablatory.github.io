import type { ModelDef } from '../types';
import gan from './gan';
import dcgan from './dcgan';

export const models: ModelDef[] = [gan, dcgan];
