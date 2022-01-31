# Copyright 2021 DeepMind Technologies Limited.
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

"""Expectations to be estimated from samples and log weights."""
import math

from annealed_flow_transport import qft_observables
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import numpy as np

ConfigDict = tp.ConfigDict
Array = tp.Array
ParticleState = tp.ParticleState


class SingleComponentMean():
  """Simple computation of the expectation of one component of the vector."""

  def __init__(self, config, num_dim: int):
    del num_dim
    self._config = config

  def __call__(self,
               samples: Array,
               log_weights: Array) -> Array:
    normalized_weights = jax.nn.softmax(log_weights)
    component_values = samples[:, self._config.component_index]
    return jnp.sum(normalized_weights * component_values)


class TwoPointSusceptibility():
  """A wrapper for the two point susceptibility observable."""

  def __init__(self, config, num_dim: int):
    self._config = config
    self._num_grid_per_dim = int(math.sqrt(num_dim))
    assert self._num_grid_per_dim ** 2 == num_dim

  def __call__(self,
               samples: Array,
               log_weights: Array) -> Array:
    num_batch = np.shape(samples)[0]
    reshaped_samples = jnp.reshape(samples, (num_batch,
                                             self._num_grid_per_dim,
                                             self._num_grid_per_dim))

    return qft_observables.estimate_two_point_susceptibility(
        reshaped_samples, log_weights, self._num_grid_per_dim)


class IsingEnergyDensity():
  """A wrapper for the Ising energy density observable."""

  def __init__(self, config, num_dim: int):
    self._config = config
    self._num_grid_per_dim = int(math.sqrt(num_dim))
    assert self._num_grid_per_dim ** 2 == num_dim

  def __call__(self,
               samples: Array,
               log_weights: Array) -> Array:
    num_batch = np.shape(samples)[0]
    reshaped_samples = jnp.reshape(samples, (num_batch,
                                             self._num_grid_per_dim,
                                             self._num_grid_per_dim))

    return qft_observables.estimate_ising_energy_density(
        reshaped_samples, log_weights)
