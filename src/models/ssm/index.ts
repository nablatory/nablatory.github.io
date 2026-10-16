import type { ModelDef } from '../types';
import linearSsm from './linear-ssm';
import mamba from './mamba';

export const models: ModelDef[] = [linearSsm, mamba];
