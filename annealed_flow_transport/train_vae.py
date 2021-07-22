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
from typing import Any, Mapping, NamedTuple, Tuple

from absl import app
from absl import flags
from absl import logging
import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from matplotlib import pylab as plt
from ml_collections.config_flags import config_flags
import numpy as np
import optax
import tensorflow_datasets as tfds


Array = jnp.ndarray
MNIST_IMAGE_SHAPE = (28, 28, 1)
Batch = Mapping[str, np.ndarray]
RandomKey = tp.RandomKey
OptState = tp.OptState
Params = Any
UpdateFn = tp.UpdateFn


def kl_divergence_standard_gaussian(mean, std) -> Array:
  """KL divergence from diagonal Gaussian with mean std to standard normal.

  Independence means the KL is a sum of KL divergences for each dimension.
  expectation_{q(x)}log prod_i q_i(x_i)/p_i(x_i)
  = sum_{i=1}^{N} expectation_{q_i(x_i)} log q_i(x_i) / p_i(x_i)
  So we have a sum of KL divergence between univariate Gaussians where
  p_i(x_i) is a standar normal.
  So each term is 0.5 * ((std)^2 + (mean)^2 - 1 - 2 ln (std) )
  Args:
    mean: Array of length (ndim,)
    std: Array of length (ndim,)
  Returns:
    KL-divergence Array of shape ().
  """
  chex.assert_rank([mean, std], [1, 1])
  terms = 0.5 * (jnp.square(std) + jnp.square(mean) - 1. - 2. * jnp.log(std))
  return jnp.sum(terms)


def batch_kl_divergence_standard_gaussian(mean, std) -> Array:
  """Mean KL divergence diagonal Gaussian with mean std to standard normal.

  Works for batches of mean/std.
  Independence means the KL is a sum of KL divergences for each dimension.
  expectation_{q(x)}log prod_i q_i(x_i)/p_i(x_i)
  = sum_{i=1}^{N} expectation_{q_i(x_i)} log q_i(x_i) / p_i(x_i)
  So we have a sum of KL divergence between univariate Gaussians where
  p_i(x_i) is a standar normal.
  So each term is 0.5 * ((std)^2 + (mean)^2 - 1 - 2 ln (std) )
  Args:
    mean: Array of length (batch,ndim)
    std: Array of length (batch,ndim)
  Returns:
    KL-divergence Array of shape ().
  """
  chex.assert_rank([mean, std], [2, 2])
  chex.assert_equal_shape([mean, std])
  batch_kls = jax.vmap(kl_divergence_standard_gaussian)(mean, std)
  return jnp.mean(batch_kls)


def generate_binarized_images(key: RandomKey, logits: Array) -> Array:
  return jax.random.bernoulli(key, jax.nn.sigmoid(logits))


def load_dataset(split: str, batch_size: int):
  """Load the dataset."""
  read_config = tfds.ReadConfig(shuffle_seed=1)
  ds = tfds.load(
      'binarized_mnist',
      split=split,
      shuffle_files=True,
      read_config=read_config)
  ds = ds.shuffle(buffer_size=10 * batch_size, seed=1)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


class VAEResult(NamedTuple):
  sample_image: jnp.ndarray
  reconst_sample: jnp.ndarray
  latent_mean: jnp.ndarray
  latent_std: jnp.ndarray
  logits: jnp.ndarray


class ConvEncoder(hk.Module):
  """A residual network encoder with mean stdev outputs."""

  def __init__(self,
               num_latents: int = 20):
    super().__init__()
    self._num_latents = num_latents

  def __call__(self, x: Array) -> Tuple[Array, Array]:
    conv_a = hk.Conv2D(kernel_shape=(4, 4),
                       stride=(2, 2),
                       output_channels=16,
                       padding='valid')
    conv_b = hk.Conv2D(kernel_shape=(4, 4),
                       stride=(2, 2),
                       output_channels=32,
                       padding='valid')
    flatten = hk.Flatten()
    sequential = hk.Sequential([conv_a,
                                jax.nn.relu,
                                conv_b,
                                jax.nn.relu,
                                flatten])
    progress = sequential(x)
    def get_output_params(progress_in, name=None):
      flat_output = hk.Linear(self._num_latents, name=name)(progress_in)
      flat_output = hk.LayerNorm(create_scale=True,
                                 create_offset=True,
                                 axis=1)(flat_output)
      return flat_output

    latent_mean = get_output_params(progress)
    unconst_std_dev = get_output_params(progress)
    latent_std = jax.nn.softplus(unconst_std_dev)

    return latent_mean, latent_std


class ConvDecoder(hk.Module):
  """A residual network decoder with logit outputs."""

  def __init__(self, image_shape: Tuple[int, int, int] = MNIST_IMAGE_SHAPE):
    super().__init__()
    self._image_shape = image_shape

  def __call__(self,
               z: Array) -> Tuple[Array, Array, Array]:
    linear_features = 7 * 7 * 32
    linear = hk.Linear(linear_features)
    progress = linear(z)
    hk.LayerNorm(create_scale=True,
                 create_offset=True,
                 axis=1)(progress)
    progress = jnp.reshape(progress, (-1, 7, 7, 32))
    deconv_a = hk.Conv2DTranspose(
        kernel_shape=(3, 3), stride=(2, 2), output_channels=64)
    deconv_b = hk.Conv2DTranspose(
        kernel_shape=(3, 3), stride=(2, 2), output_channels=32)
    deconv_c = hk.Conv2DTranspose(
        kernel_shape=(3, 3), stride=(1, 1), output_channels=1)
    sequential = hk.Sequential([deconv_a,
                                jax.nn.relu,
                                deconv_b,
                                jax.nn.relu,
                                deconv_c])
    progress = sequential(progress)
    return progress


class ConvVAE(hk.Module):
  """A VAE with residual nets, diagonal normal q and logistic mixture output."""

  def __init__(self, num_latents: int = 30,
               output_shape: Tuple[int, int, int] = MNIST_IMAGE_SHAPE):
    super().__init__()
    self._num_latents = num_latents
    self._output_shape = output_shape
    self.encoder = ConvEncoder(self._num_latents)
    self.decoder = ConvDecoder()

  def __call__(self, x: Array) -> VAEResult:
    x = x.astype(jnp.float32)
    latent_mean, latent_std = self.encoder(x)
    latent = latent_mean + latent_std * jax.random.normal(
        hk.next_rng_key(), latent_mean.shape)
    free_latent = jax.random.normal(hk.next_rng_key(), latent_mean.shape)
    logits = self.decoder(latent)
    free_logits = self.decoder(free_latent)
    reconst_sample = jax.nn.sigmoid(logits)
    sample_image = jax.nn.sigmoid(free_logits)
    return VAEResult(sample_image, reconst_sample, latent_mean, latent_std,
                     logits)


def binary_cross_entropy_from_logits(logits: Array, labels: Array) -> Array:
  """Numerically stable implementation of binary cross entropy with logits.

  For an individual term we follow a standard manipulation of the loss:
  H = -label * log sigmoid(logit) - (1-label) * log (1-sigmoid(logit))
  = logit - label * logit + log(1+exp(-logit))
  or for logit < 0 we take a different version for numerical stability.
  = - label * logit + log(1+exp(logit))
  combining to avoid a conditional.
  = max(logit, 0) - label * logit + log(1+exp(-abs(logit)))

  Args:
    logits: (batch, sample_shape) containing logits of class probs.
    labels: (batch, sample_shape) containing {0, 1} class labels.
  Returns:
    sum of loss over all shape indices then mean of loss over batch index.
  """
  chex.assert_equal_shape([logits, labels])
  max_logits_zero = jax.nn.relu(logits)
  negative_abs_logits = -jnp.abs(logits)
  terms = max_logits_zero - logits*labels + jax.nn.softplus(negative_abs_logits)
  return jnp.sum(jnp.mean(terms, axis=0))


def vae_loss(target: Array, logits: Array, latent_mean: Array,
             latent_std: Array) -> Array:
  log_loss = binary_cross_entropy_from_logits(logits, target)
  kl_term = batch_kl_divergence_standard_gaussian(latent_mean, latent_std)
  free_energy = log_loss + kl_term
  return free_energy


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
    loss = vae_loss(batch['image'], output.logits, output.latent_mean,
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
  train_dataset = load_dataset(tfds.Split.TRAIN, batch_size)
  validation_dataset = load_dataset(tfds.Split.TEST, batch_size)

  def call_vae(x):
    res_vae = ConvVAE(num_latents=num_latents)
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
    loss = vae_loss(curr_batch['image'], output.logits,
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
