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

"""Tests for annealed_flow_transport.flow_transport."""

import hashlib
import os.path

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport.densities import AutoEncoderLikelihood
from annealed_flow_transport.densities import ChallengingTwoDimensionalMixture
from annealed_flow_transport.densities import FunnelDistribution
from annealed_flow_transport.densities import MultivariateNormalDistribution
from annealed_flow_transport.densities import NormalDistribution
from annealed_flow_transport.densities import phi_four_log_density
from annealed_flow_transport.densities import PhiFourTheory

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import ml_collections
import numpy as np

ConfigDict = ml_collections.ConfigDict

join = os.path.join
dirname = os.path.dirname


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


def get_normal_config():
  config = ConfigDict()
  config.loc = 1.
  config.scale = 1.
  return config


def get_multivariate_normal_config():
  config = ConfigDict()
  config.shared_mean = 1.
  config.diagonal_cov = 1.
  return config


class BasicShapesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Normal', NormalDistribution, 1, get_normal_config()),
      ('MultivariateNormal', MultivariateNormalDistribution, 2,
       get_multivariate_normal_config()),
      ('TwoDimensionalMixture', ChallengingTwoDimensionalMixture, 2, None),
      ('FunnelDistribution', FunnelDistribution, 10, None),)
  def test_shapes(self, test_class, num_dim: int, config):
    num_batch = 7
    test_matrix = jnp.arange(num_dim * num_batch).reshape((num_batch, num_dim))
    if not config:
      config = ConfigDict()
    test_density = test_class(config, num_dim)
    output_log_densities = test_density(test_matrix)
    self.assertEqual(output_log_densities.shape, (num_batch,))


class VAETest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('First digit', 0, '9ea2704b4fafa24f97c5e330506bd2c9'),
      ('Tenth digit', 9, '67a03001cd0eadedff8300f2b6cb7f03'),
      ('Main paper digit', 3689, 'c4ae91223d22b0ed227b401d18237e65'),)
  def test_digit_ordering(self, digit_index, target_md5_hash):
    """Confirm the digit determined by index has not changed using a hash."""
    config = ConfigDict()
    config.params_filename = join(dirname(__file__), 'data/vae.pickle')
    config.image_index = digit_index
    vae_density = AutoEncoderLikelihood(config, 30)
    hex_digit_hash = hashlib.md5(vae_density._test_image).hexdigest()
    self.assertEqual(hex_digit_hash, target_md5_hash)

  def test_density_shape(self):
    num_batch = 7
    num_dim = 30
    total_points = num_batch * num_dim
    config = ConfigDict()
    config.params_filename = join(dirname(__file__), 'data/vae.pickle')
    config.image_index = 0
    vae_density = AutoEncoderLikelihood(config, num_dim)
    test_input = (jnp.arange(total_points).reshape(num_batch, num_dim) -
                  100.) / 100.
    test_output = vae_density(test_input)
    self.assertEqual(test_output.shape, (num_batch,))

  def test_log_prior(self):
    num_dim = 30
    config = ConfigDict()
    config.params_filename = join(dirname(__file__), 'data/vae.pickle')
    config.image_index = 0
    vae_density = AutoEncoderLikelihood(config, num_dim)
    test_input = (jnp.arange(num_dim)-num_dim)/num_dim
    test_output = vae_density.log_prior(test_input)
    reference_output = jnp.sum(jax.vmap(norm.logpdf)(test_input))
    _assert_equal_vec(self, reference_output, test_output)

  def test_batching_consistency(self):
    """Paranoid test to check there is nothing wrong with batching/averages."""
    num_batch = 7
    num_dim = 30
    total_points = num_batch * num_dim
    config = ConfigDict()
    config.params_filename = join(dirname(__file__), 'data/vae.pickle')
    config.image_index = 0
    vae_density = AutoEncoderLikelihood(config, num_dim)
    test_input = (jnp.arange(total_points).reshape(num_batch, num_dim) -
                  100.) / 100.
    test_output = vae_density(test_input)
    for batch_index in range(num_batch):
      current_log_density = vae_density.total_log_probability(
          test_input[batch_index, :])
      _assert_equal_vec(self, test_output[batch_index], current_log_density)


class PhiFourTest(parameterized.TestCase):

  def test_batched_configurable(self):
    config = ConfigDict()
    config.mass_squared = -4.
    config.bare_coupling = 5.1
    batch_size = 7
    num_dim = 16
    trial_values = jnp.linspace(-2., 2., batch_size * num_dim).reshape(
        (batch_size, num_dim))
    log_density = PhiFourTheory(config, num_dim)
    log_density_val = log_density(trial_values)
    self.assertTrue(log_density_val.shape, (batch_size))

  def test_zero(self):
    lattice_shape = (8, 6)
    trial_lattice_values = jnp.zeros(lattice_shape)
    mass_squared = -4.
    bare_coupling = 5.1
    trial_log_density = phi_four_log_density(trial_lattice_values,
                                             mass_squared,
                                             bare_coupling)
    _assert_equal_vec(self, trial_log_density, 0.)

  def test_reflection_symmetry(self):
    lattice_shape = (8, 6)
    lattice_size = np.prod(lattice_shape)
    trial_lattice_values = jnp.linspace(-2., 2.,
                                        lattice_size).reshape(lattice_shape)
    reflected_trial_lattice_values = -1.*trial_lattice_values
    mass_squared = -4.
    bare_coupling = 5.1
    trial_log_density = phi_four_log_density(trial_lattice_values,
                                             mass_squared,
                                             bare_coupling)
    reflected_trial_log_density = phi_four_log_density(
        reflected_trial_lattice_values, mass_squared, bare_coupling)
    _assert_equal_vec(self, trial_log_density, reflected_trial_log_density)

  def test_translation_symmetry(self):
    lattice_shape = (8, 6)
    lattice_size = np.prod(lattice_shape)
    trial_lattice_values = jnp.linspace(-2., 2.,
                                        lattice_size).reshape(lattice_shape)
    translated_lattice_values = jnp.roll(trial_lattice_values, shift=(1, 2),
                                         axis=(0, 1))
    mass_squared = -4.
    bare_coupling = 5.1
    trial_log_density = phi_four_log_density(trial_lattice_values,
                                             mass_squared,
                                             bare_coupling)
    translated_log_density = phi_four_log_density(translated_lattice_values,
                                                  mass_squared,
                                                  bare_coupling)
    _assert_equal_vec(self, trial_log_density, translated_log_density)

if __name__ == '__main__':
  absltest.main()
