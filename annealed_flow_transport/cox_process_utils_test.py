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

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport import cox_process_utils
import jax.numpy as jnp
import numpy as np


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class CoxProcessUtilsTest(parameterized.TestCase):

  def test_get_bin_vals(self):
    bins_per_dim = 30
    bin_vals = cox_process_utils.get_bin_vals(bins_per_dim)
    self.assertEqual(bin_vals.shape, (bins_per_dim*bins_per_dim, 2))
    first_bin_vals = bin_vals[0, :]
    self.assertEqual(list(first_bin_vals), [0, 0])
    second_bin_vals = bin_vals[1, :]
    self.assertEqual(list(second_bin_vals), [0, 1])
    final_bin_vals = bin_vals[-1, :]
    self.assertEqual(list(final_bin_vals),
                     [bins_per_dim-1, bins_per_dim-1])

  def test_whites_and_latents(self):
    lower_triangular_matrix = jnp.array([[1., 0.], [-1., 2.]])
    constant_mean = 1.7
    latents = jnp.array([5.5, 3.6])
    test_white = cox_process_utils.get_white_from_latents(
        latents, constant_mean, lower_triangular_matrix)
    test_latents = cox_process_utils.get_latents_from_white(
        test_white, constant_mean, lower_triangular_matrix)
    _assert_equal_vec(self, latents, test_latents)

  def test_gram(self):
    def pairwise_function(x, y):
      return jnp.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    test_points = jnp.array([[1., 2.], [5.2, 6.1], [7.2, 3.6]])
    test_gram_matrix = cox_process_utils.gram(pairwise_function,
                                              test_points)
    validation_gram_matrix = np.zeros((3, 3))
    for row_index in range(3):
      for col_index in range(3):
        pair_val = pairwise_function(test_points[row_index, :],
                                     test_points[col_index, :])
        validation_gram_matrix[row_index, col_index] = pair_val

    _assert_equal_vec(self, test_gram_matrix, validation_gram_matrix)

  def test_bin_counts(self):
    num_bins_per_dim = 2
    test_array = jnp.array([[0.25, 0.25],  # in bin [0, 0]
                            [0.75, 0.75],  # in bin [1, 1]
                            [0.0, 0.0],  # in bin [0, 0]
                            [0.0, 1.0],  # in bin [0, 1] an edge case
                            [1.0, 1.0],  # in bin [1, 1] a corner case
                            [0.22, 0.22]])  # in bin [0, 0]
    test_bin_counts = cox_process_utils.get_bin_counts(test_array,
                                                       num_bins_per_dim)
    validation_bin_counts = jnp.array([[3, 1], [0, 2]])

    _assert_equal_vec(self, test_bin_counts, validation_bin_counts)

if __name__ == '__main__':
  absltest.main()
