import type { ModelDef } from '../types';
import rssm from './rssm';
import dreamer from './dreamer';
import tdmpc from './tdmpc';

export const models: ModelDef[] = [rssm, dreamer, tdmpc];
