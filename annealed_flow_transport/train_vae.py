# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a convolutional variational autoencoder for likelihood experiments.

Some Jax/Haiku programming idioms inspired by OSS Apache 2.0 Haiku vae example.

A pretrained version of this model is already included in data/vae.pickle.
To run one of the sampling algorithms on that trained model use configs/vae.py
This training script is included for full reproducibility.
"""
import os
import pickle
import time
from typing import Any, Tuple

from absl import app
from absl import flags
from absl import logging
from annealed_flow_transport import vae
import annealed_flow_transport.aft_types as tp
import haiku as hk
import jax
from matplotlib import pylab as plt
from ml_collections.config_flags import config_flags
import optax
import tensorflow_datasets as tfds


Array = tp.Array
Batch = tp.VaeBatch
MNIST_IMAGE_SHAPE = tp.MNIST_IMAGE_SHAPE
RandomKey = tp.RandomKey
OptState = tp.OptState
Params = Any
UpdateFn = tp.UpdateFn
VAEResult = tp.VAEResult


def train_step(vae_apply,
               vae_params,
               opt_state: OptState,
               opt_update: UpdateFn,
               batch: Batch,
               random_key: RandomKey) -> Tuple[Params, OptState, Array]:
  """A single step of training for the VAE."""
  def params_loss(loc_params):
    scaled_image = batch['image']
    output: VAEResult = vae_apply(loc_params, random_key, scaled_image)
    loss = vae.vae_loss(batch['image'], output.logits, output.latent_mean,
                        output.latent_std)
    return loss
  value, grads = jax.value_and_grad(params_loss)(vae_params)
  updates, new_opt_state = opt_update(grads, opt_state)
  new_params = optax.apply_updates(vae_params, updates)
  return new_params, new_opt_state, value


def save_image(reconst_image: Array, train_image: Array, sample_image: Array,
               num_plot: int, opt_iter: int, output_directory: str):
  """Show image plots."""
  overall_size = 1.5
  unused_fig, axes = plt.subplots(
      3, num_plot, figsize=(overall_size*num_plot, overall_size*3))
  def plot_image(curr_plot_index, sub_index, data):
    axes[sub_index, curr_plot_index].imshow(data, cmap='gray', vmin=0., vmax=1.)
  for plot_index in range(num_plot):
    for (sub_index,
         datum) in zip(range(3), (train_image, reconst_image, sample_image)):
      plot_image(plot_index, sub_index, datum[plot_index, :, :, 0])
  plt.savefig(output_directory+str(opt_iter)+'.png')
  plt.close()


def train_vae(batch_size: int,
              num_latents: int,
              random_seed: int,
              step_size: float,
              output_dir_stub,
              train_iters: int,
              report_period: int
              ):
  """Train the VAE on binarized MNIST.

  Args:
    batch_size: Batch size for training and validation.
    num_latents: Number of latents for VAE latent space.
    random_seed: Random seed for training.
    step_size: Step size for ADAM optimizer.
    output_dir_stub: Where to store files if truthy otherwise don't store files.
    train_iters: Number of iterations to run training for.
    report_period: Period between reporting losses and storing files.
  """
  train_dataset = vae.load_dataset(tfds.Split.TRAIN, batch_size)
  validation_dataset = vae.load_dataset(tfds.Split.TEST, batch_size)

  def call_vae(x):
    res_vae = vae.ConvVAE(num_latents=num_latents)
    return res_vae(x)
  vae_fn = hk.transform(call_vae)
  rng_seq = hk.PRNGSequence(random_seed)

  vae_params = vae_fn.init(
      next(rng_seq), next(train_dataset)['image'])
  opt = optax.chain(
      optax.clip_by_global_norm(1e5),
      optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
      optax.scale(-step_size)
  )
  opt_state = opt.init(vae_params)
  opt_update = opt.update

  def train_step_short(curr_params, curr_opt_state, curr_batch, curr_key):
    return train_step(vae_fn.apply, curr_params, curr_opt_state, opt_update,
                      curr_batch, curr_key)

  def compute_validation(curr_params, curr_batch, curr_key):
    output: VAEResult = vae_fn.apply(curr_params, curr_key, curr_batch['image'])
    loss = vae.vae_loss(curr_batch['image'], output.logits,
                        output.latent_mean, output.latent_std)
    return loss, output.reconst_sample, curr_batch['image'], output.sample_image

  train_step_jit = jax.jit(train_step_short)
  compute_validation_jit = jax.jit(compute_validation)

  if output_dir_stub:
    output_directory = output_dir_stub + time.strftime('%a_%d_%b_%Y_%H:%M:%S/',
                                                       time.gmtime())
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

  for opt_iter in range(train_iters):
    vae_params, opt_state, train_loss = train_step_jit(
        vae_params, opt_state, next(train_dataset), next(rng_seq))
    if opt_iter % report_period == 0:
      validation_loss, reconst_sample, example, sample = compute_validation_jit(
          vae_params, next(validation_dataset), next(rng_seq))
      logging.info('Step: %5d: Training VFE: %.3f', opt_iter, train_loss)
      logging.info('Step: %5d: Validation VFE: %.3f', opt_iter, validation_loss)
      if output_dir_stub:
        save_image(reconst_sample, example, sample, 8, opt_iter,
                   output_directory)

  if output_dir_stub:
    save_result(vae_params, output_directory)


def get_checkpoint_filename():
  return 'vae.pickle'


def save_result(state, output_directory: str):
  ckpt_filename = os.path.join(output_directory, get_checkpoint_filename())
  with open(ckpt_filename, 'wb') as f:
    pickle.dump(state, f)


def main(argv):
  config = FLAGS.train_vae_config
  info = 'Displaying config '+str(config)
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train_vae(
      batch_size=config.batch_size,
      num_latents=config.num_latents,
      random_seed=config.random_seed,
      step_size=config.step_size,
      output_dir_stub=config.output_dir_stub,
      train_iters=config.train_iters,
      report_period=config.report_period)


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  config_flags.DEFINE_config_file('train_vae_config',
                                  './train_vae_configs/vae_config.py',
                                  'VAE training configuration.')
  app.run(main)
