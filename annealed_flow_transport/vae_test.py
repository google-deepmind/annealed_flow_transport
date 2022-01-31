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

"""Tests for annealed_flow_transport.train_vae."""

from absl.testing import parameterized
from annealed_flow_transport import vae
import jax
import jax.numpy as jnp


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class TestKLDivergence(parameterized.TestCase):

  def reference_kl_divergence(self, mean, std):

    def termwise_kl(single_mean, single_std):
      term_a = -jnp.log(single_std)
      term_b = 0.5 * single_std**2
      term_c = 0.5 * single_mean**2
      term_d = -0.5
      return term_a + term_b + term_c + term_d

    return jnp.sum(jax.vmap(termwise_kl)(mean, std))

  def test_kl_divergence(self):
    num_dim = 4
    mean = jnp.zeros(num_dim)
    std = jnp.ones(num_dim)
    test_kl = vae.kl_divergence_standard_gaussian(mean, std)
    _assert_equal_vec(self, test_kl, 0.)

    mean_b = jnp.arange(4)
    std_b = jnp.array([1.3, 1.7, 1.8, 2.0])

    test_kl_b = vae.kl_divergence_standard_gaussian(mean_b, std_b)
    reference_kl_b = self.reference_kl_divergence(mean_b, std_b)
    _assert_equal_vec(self, test_kl_b, reference_kl_b)

  def test_batch_kl_divergence(self):
    num_dim = 5
    num_batch = 3
    total_points = num_dim * num_batch
    means = jnp.arange(total_points).reshape((num_batch, num_dim))
    stds = jnp.arange(total_points).reshape((num_batch, num_dim))+1.5

    total_reference_kl_divergence = 0.
    for batch_index in range(num_batch):
      total_reference_kl_divergence += self.reference_kl_divergence(
          means[batch_index], stds[batch_index])

    reference_mean_kl = total_reference_kl_divergence/num_batch
    test_mean_kl = vae.batch_kl_divergence_standard_gaussian(means,
                                                             stds)
    _assert_equal_vec(self, reference_mean_kl, test_mean_kl)


class TestBinaryCrossEntropy(parameterized.TestCase):

  def reference_binary_cross_entropy(self, logits, labels):

    def single_binary_cross_entropy(logit, label):
      h = label * jax.nn.softplus(-logit) + (1 - label) * jax.nn.softplus(logit)
      return h
    accumulator = 0.
    (num_batch, num_dim_a, num_dim_b) = logits.shape
    for batch_index in range(num_batch):
      for dim_a in range(num_dim_a):
        for dim_b in range(num_dim_b):
          accumulator += single_binary_cross_entropy(
              logits[batch_index, dim_a, dim_b], labels[batch_index, dim_a,
                                                        dim_b])
    return accumulator/num_batch

  def test_binary_cross_entropy(self):
    num_batch = 7
    num_pixel_per_image_dim = 3
    total_elements = num_batch * num_pixel_per_image_dim * num_pixel_per_image_dim

    trial_logits = jnp.arange(total_elements).reshape(
        (num_batch, num_pixel_per_image_dim, num_pixel_per_image_dim)) - 10.
    sequence = jnp.arange(total_elements).reshape(
        (num_batch, num_pixel_per_image_dim, num_pixel_per_image_dim))
    trial_labels = jnp.mod(sequence, 2)

    test_loss = vae.binary_cross_entropy_from_logits(trial_logits,
                                                     trial_labels)
    reference_loss = self.reference_binary_cross_entropy(trial_logits,
                                                         trial_labels)
    _assert_equal_vec(self, test_loss, reference_loss)
