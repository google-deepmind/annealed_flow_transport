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

"""Code for cox process density utilities."""

import itertools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
import numpy as np

# TypeDefs
NpArray = np.ndarray
Array = jnp.ndarray


def get_bin_counts(array_in: NpArray,
                   num_bins_per_dim: int) -> NpArray:
  """Divide two dimensional input space into a grid and count points in each.

  Point on the upper edge, which does happen in the data, go into the lower bin.
  The occurrence of these points is an artefact of the rescaling done on data.

  Args:
    array_in: (num_points,2) containing points in square [0,1]^2
    num_bins_per_dim: the number of bins per dimension for the grid.

  Returns:
    Numpy array of shape containing (num_bins_per_dim, num_bins_per_dim) counts.
  """
  chex.assert_rank(array_in, 2)
  scaled_array = array_in * num_bins_per_dim
  counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
  for elem in scaled_array:
    flt_row, col_row = np.floor(elem)
    row = int(flt_row)
    col = int(col_row)
    # Deal with the case where the point lies exactly on upper/rightmost edge.
    if row == num_bins_per_dim:
      row -= 1
    if col == num_bins_per_dim:
      col -= 1
    counts[row, col] += 1
  return counts


def get_bin_vals(num_bins: int) -> NpArray:
  grid_indices = jnp.arange(num_bins)
  bin_vals = jnp.array([
      jnp.array(elem) for elem in itertools.product(grid_indices, grid_indices)
  ])
  return bin_vals


def gram(kernel, xs: Array) -> Array:
  """Given a kernel function and an array of points compute a gram matrix."""
  return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(xs))(xs)


def kernel_func(x: Array,
                y: Array,
                signal_variance: Array,
                num_grid_per_dim: int,
                raw_length_scale: Array) -> Array:
  """Compute covariance/kernel function.

  K(m,n) = signal_variance * exp(-|m-n|/(num_grid_per_dim*raw_length_scale))

  Args:
    x: First point shape (num_spatial_dim,)
    y: Second point shape (num_spatial_dim,)
    signal_variance: non-negative scalar.
    num_grid_per_dim: Number of grid points per spatial dimension.
    raw_length_scale: Length scale of the undiscretized process.

  Returns:
    Scalar value of covariance function.
  """
  chex.assert_equal_shape([x, y])
  chex.assert_rank(x, 1)
  normalized_distance = jnp.linalg.norm(x - y, 2) / (
      num_grid_per_dim * raw_length_scale)
  return signal_variance * jnp.exp(-normalized_distance)


def poisson_process_log_likelihood(latent_function: Array,
                                   bin_area: Array,
                                   flat_bin_counts: Array) -> Array:
  """Discretized Poisson process log likelihood.

  Args:
    latent_function: Intensity per unit area of shape (total_dimensions,)
    bin_area: Scalar bin_area.
    flat_bin_counts: Non negative integer counts of shape (total_dimensions,)

  Returns:
    Total log likelihood of points.
  """
  chex.assert_rank([latent_function, bin_area], [1, 0])
  chex.assert_equal_shape([latent_function, flat_bin_counts])
  first_term = latent_function * flat_bin_counts
  second_term = -bin_area * jnp.exp(latent_function)
  return jnp.sum(first_term+second_term)


def get_latents_from_white(white: Array, const_mean: Array,
                           cholesky_gram: Array) -> Array:
  """Get latents from whitened representation.

  Let f = L e + mu where e is distributed as standard multivariate normal.
  Then Cov[f] = LL^T .
  In the present case L is assumed to be lower triangular and is given by
  the input cholesky_gram.
  mu_zero is a constant so that mu_i = const_mean for all i.

  Args:
    white: shape (total_dimensions,) e.g. (900,) for a 30x30 grid.
    const_mean: scalar.
    cholesky_gram: shape (total_dimensions, total_dimensions)

  Returns:
    points in the whitened space of shape (total_dimensions,)
  """
  chex.assert_rank([white, const_mean, cholesky_gram], [1, 0, 2])
  latent_function = jnp.matmul(cholesky_gram, white) + const_mean
  chex.assert_equal_shape([latent_function, white])
  return latent_function


def get_white_from_latents(latents: Array,
                           const_mean: Array,
                           cholesky_gram: Array) -> Array:
  """Get whitened representation from function representation.

  Let f = L e + mu where e is distributed as standard multivariate normal.
  Then Cov[f] = LL^T and e = L^-1(f-mu).
  In the present case L is assumed to be lower triangular and is given by
  the input cholesky_gram.
  mu_zero is a constant so that mu_i = const_mean for all i.

  Args:
    latents: shape (total_dimensions,) e.g. (900,) for a 30x30 grid.
    const_mean: scalar.
    cholesky_gram: shape (total_dimensions, total_dimensions)

  Returns:
    points in the whitened space of shape (total_dimensions,)
  """
  chex.assert_rank([latents, const_mean, cholesky_gram], [1, 0, 2])
  white = slinalg.solve_triangular(
      cholesky_gram, latents - const_mean, lower=True)
  chex.assert_equal_shape([latents, white])
  return white
