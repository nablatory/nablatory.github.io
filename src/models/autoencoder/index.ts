import type { ModelDef } from '../types';
import autoencoder from './autoencoder';
import denoising from './denoising';
import vae from './vae';

export const models: ModelDef[] = [autoencoder, denoising, vae];
