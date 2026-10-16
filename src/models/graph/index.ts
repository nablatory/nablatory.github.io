import type { ModelDef } from '../types';
import gcn from './gcn';
import graphsage from './graphsage';
import gat from './gat';

export const models: ModelDef[] = [gcn, graphsage, gat];
