import type { FamilyId, FamilyInfo, ModelDef } from './types';
import { models as feedforward } from './feedforward';
import { models as convolutional } from './convolutional';
import { models as recurrent } from './recurrent';
import { models as transformer } from './transformer';
import { models as autoencoder } from './autoencoder';
import { models as diffusion } from './diffusion';
import { models as graph } from './graph';
import { models as gan } from './gan';
import { models as worldModel } from './world-model';
import { models as ssm } from './ssm';
import { models as neuralDynamics } from './neural-dynamics';

export const FAMILIES: FamilyInfo[] = [
  {
    id: 'feedforward',
    name: 'Feedforward',
    summary: 'Compositions of affine maps and pointwise nonlinearities.',
  },
  {
    id: 'convolutional',
    name: 'Convolutional',
    summary: 'Shared local filters slid across a spatial grid.',
  },
  {
    id: 'recurrent',
    name: 'Recurrent',
    summary: 'A hidden state updated by the same cell at every time step.',
  },
  {
    id: 'transformer',
    name: 'Transformers',
    summary: 'Tokens mixed by softmax attention, then transformed position-wise.',
  },
  {
    id: 'autoencoder',
    name: 'Autoencoders',
    summary: 'An encoder to a low-dimensional code and a decoder back.',
  },
  {
    id: 'diffusion',
    name: 'Diffusion',
    summary: 'Data corrupted by Gaussian noise and recovered by a learned denoiser.',
  },
  {
    id: 'graph',
    name: 'Graph neural networks',
    summary: 'Node features updated by aggregating over neighbours.',
  },
  {
    id: 'gan',
    name: 'GANs',
    summary: 'A generator and a discriminator trained against each other.',
  },
  {
    id: 'world-model',
    name: 'World models',
    summary: 'Learned latent dynamics used to predict and plan.',
  },
  {
    id: 'ssm',
    name: 'State space models',
    summary: 'Linear recurrences from discretised continuous-time systems.',
  },
  {
    id: 'neural-dynamics',
    name: 'Neural dynamics',
    summary: 'Hidden states defined by learned differential equations.',
  },
];

const byFamily: Record<FamilyId, ModelDef[]> = {
  feedforward,
  convolutional,
  recurrent,
  transformer,
  autoencoder,
  diffusion,
  graph,
  gan,
  'world-model': worldModel,
  ssm,
  'neural-dynamics': neuralDynamics,
};

export const MODELS: ModelDef[] = FAMILIES.flatMap((f) => byFamily[f.id]);

export function modelsOf(family: FamilyId): ModelDef[] {
  return byFamily[family];
}

export function findModel(id: string): ModelDef | undefined {
  return MODELS.find((m) => m.id === id);
}
