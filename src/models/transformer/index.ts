import type { ModelDef } from '../types';
import transformer from './transformer';
import bert from './bert';
import gpt from './gpt';
import vit from './vit';

export const models: ModelDef[] = [transformer, bert, gpt, vit];
