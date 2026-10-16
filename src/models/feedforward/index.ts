import type { ModelDef } from '../types';
import perceptron from './perceptron';
import mlp from './mlp';
import resmlp from './resmlp';

export const models: ModelDef[] = [perceptron, mlp, resmlp];
