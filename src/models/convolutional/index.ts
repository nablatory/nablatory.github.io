import type { ModelDef } from '../types';
import lenet5 from './lenet5';
import vgg from './vgg';
import resnet from './resnet';
import unet from './unet';

export const models: ModelDef[] = [lenet5, vgg, resnet, unet];
