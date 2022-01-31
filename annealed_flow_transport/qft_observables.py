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

"""Quantum field theory (QFT) observables, particularly phi^four theory."""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


def estimate_two_point_green(offset_x: Tuple[int, int],
                             samples: Array,
                             log_weights: Array) -> Array:
  """Estimate the connected two point Green's function from weighted samples.

  This equals 1/V sum_y ( <phi(y) phi(y+x) > - <phi(y)><phi(y+x)>).
  Where V is the lattice volume. phi are the values of the field on the lattice.
  For more details see:

  Equation 22. Albergo, Kanwar and Shanahan (2019) Phys. Rev. D
    ''Flow-based generative models for Markov chain Monte Carlo
     in lattice field theory.''

  Args:
    offset_x: 2-tuple containing lattice offset x.
    samples: Array of size (num_batch, L_x, L_y)- particle values on 2D lattice.
    log_weights: Array of size (num_batch,) - particle log weights.
  Returns:
    Scalar estimate of two point greens function at offset_x.
  """
  chex.assert_rank([samples, log_weights], [3, 1])
  offset_samples = jnp.roll(samples, shift=offset_x, axis=(1, 2))
  normalized_log_weights = jax.nn.softmax(log_weights)
  # In this case means are all taken to be zero by symmetry.
  covariance = jnp.sum(
      normalized_log_weights[:, None, None] * samples * offset_samples, axis=0)
  # take spatial mean 1/V sum_y ...
  two_point_green = jnp.mean(covariance)
  return two_point_green


def estimate_zero_momentum_green(samples: Array,
                                 log_weights: Array,
                                 time_offset: int) -> Array:
  """Estimate the momentum space two point Green's function at momentum zero.

  We adopt the convention that the first grid axis corresponds to space.
  Usually it doesn't matter which one you choose along as you are consistent.
  It is important not to mix up lattice sizes when they are unequal.

  For more details see:

  Equation 23. Albergo, Kanwar and Shanahan (2019) Phys. Rev. D
    ''Flow-based generative models for Markov chain Monte Carlo
     in lattice field theory.''

  Args:
    samples: Array of size (num_batch, L_x, L_y)- particle values on 2D lattice.
    log_weights: Array of size (num_batch,) - particle log weights.
    time_offset: Offset in lattice units in the time dimension.
  Returns:
    Scalar estimate of the zero momentum Green's function.
  """
  chex.assert_rank([samples, log_weights], [3, 1])
  num_space_indices = np.shape(samples)[2]
  offset_indices = [(elem, time_offset) for elem in range(num_space_indices)]
  # The complex exponential term is 1 because of zero momentum assumption.
  running_total = 0.
  for offset in offset_indices:
    running_total += estimate_two_point_green(offset, samples, log_weights)
  return running_total/num_space_indices


def estimate_time_vals(samples: Array,
                       log_weights: Array,
                       num_time_indices: int) -> Array:
  """Estimate zero momentum Green for a range of different time offsets."""
  time_vals = np.zeros(num_time_indices)
  for time_offset in range(num_time_indices):
    time_vals[time_offset] = estimate_zero_momentum_green(samples,
                                                          log_weights,
                                                          time_offset)
  return time_vals


def estimate_two_point_susceptibility(samples: Array,
                                      log_weights: Array,
                                      num_grid_per_dim: int) -> Array:
  """Estimate the two point susceptibility."""
  total = 0.
  for row_index in range(num_grid_per_dim):
    for col_index in range(num_grid_per_dim):
      offset = (row_index, col_index)
      total += estimate_two_point_green(offset, samples, log_weights)
  return total


def estimate_ising_energy_density(samples: Array,
                                  log_weights: Array) -> Array:
  """Estimate Ising energy density."""
  total = 0.
  unit_displacements = [(0, 1), (1, 0)]
  for offset in unit_displacements:
    total += estimate_two_point_green(offset, samples, log_weights)
  return total/len(unit_displacements)
