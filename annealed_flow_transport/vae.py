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

"""Convolutional variational autoencoder code for likelihood experiments.

Some Jax/Haiku programming idioms inspired by OSS Apache 2.0 Haiku vae example.

A pretrained version of this model is already included in data/vae.pickle.
To run one of the sampling algorithms on that trained model use configs/vae.py
This training script is included for full reproducibility.
"""
from typing import Any, Tuple

import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds


Array = tp.Array
MNIST_IMAGE_SHAPE = tp.MNIST_IMAGE_SHAPE
Batch = tp.VaeBatch
RandomKey = tp.RandomKey
OptState = tp.OptState
Params = Any
UpdateFn = tp.UpdateFn
VAEResult = tp.VAEResult


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
