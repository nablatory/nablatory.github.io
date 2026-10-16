import type { ModelDef } from '../types';
import ddpm from './ddpm';
import ddim from './ddim';

export const models: ModelDef[] = [ddpm, ddim];
