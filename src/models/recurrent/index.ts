import type { ModelDef } from '../types';
import rnn from './rnn';
import lstm from './lstm';
import gru from './gru';

export const models: ModelDef[] = [rnn, lstm, gru];
