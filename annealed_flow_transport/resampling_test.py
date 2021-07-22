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

"""Tests for annealed_flow_transport.resampling."""

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport import resampling
import jax
import jax.numpy as jnp


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class ResamplingTest(parameterized.TestCase):

  def test_ess(self):
    # Test equal unnormalized weights come out correctly.
    num_samples = 32
    arbitrary_number = -3.7
    log_weights = -arbitrary_number*jnp.ones(num_samples)

    test_log_ess = resampling.log_effective_sample_size(log_weights)
    true_log_ess = jnp.log(num_samples)

    _assert_equal_vec(self, test_log_ess, true_log_ess)

    # Test an arbitrary simple case.
    weights = jnp.array([0.3, 0.2, 0.5])
    true_log_ess_b = jnp.log(1./jnp.sum(weights**2))

    log_weights_b = jnp.log(weights) + arbitrary_number
    test_log_ess_b = resampling.log_effective_sample_size(log_weights_b)

    _assert_equal_vec(self, true_log_ess_b, test_log_ess_b)

  def test_simple_resampling(self):
    arbitrary_number = -5.2
    num_samples = 10000

    key = jax.random.PRNGKey(1)
    # First we sample from normal distribution d
    # and then perform an importance correction to normal distribution a.
    dimension = 1
    key, subkey = jax.random.split(key, 2)
    samples = jax.random.normal(subkey, shape=(num_samples, dimension))
    weights = jnp.array([0.5, 0.25, 0.25]+[0.]*(num_samples-3))
    log_unnormalized_weights = jnp.log(weights)+arbitrary_number

    target_mean = 0.5*samples[0] + 0.25*samples[1] + 0.25*samples[2]
    target_variance = 0.5 * (samples[0] - target_mean)**2 + 0.25 * (
        samples[1] - target_mean)**2 + 0.25 * (samples[2] - target_mean)**2
    target_weights = -1.*jnp.ones(num_samples)*jnp.log(num_samples)

    resampled, log_weights_new = resampling.simple_resampling(
        key, log_unnormalized_weights, samples)
    empirical_mean = jnp.mean(resampled)
    empirical_variance = jnp.var(resampled)

    _assert_equal_vec(self, empirical_mean, target_mean, atol=1e-2)
    _assert_equal_vec(self, empirical_variance, target_variance, atol=1e-2)
    _assert_equal_vec(self, log_weights_new, target_weights)

  def test_optionally_resample(self):
    num_samples = 100
    dimension = 2
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key, 2)
    samples = jax.random.normal(subkey, shape=(num_samples, dimension))
    key, subkey = jax.random.split(key, 2)
    log_weights = jax.random.normal(subkey, shape=(num_samples,))
    log_ess = resampling.log_effective_sample_size(log_weights)
    resamples, log_uniform_weights = resampling.simple_resampling(key,
                                                                  log_weights,
                                                                  samples)
    threshold_lower = 0.9/num_samples*jnp.exp(log_ess)
    threshold_upper = 1.1/num_samples*jnp.exp(log_ess)
    should_be_samples, should_be_log_weights = resampling.optionally_resample(
        key, log_weights, samples, threshold_lower)
    should_be_resamples, should_be_uniform_weights = resampling.optionally_resample(
        key, log_weights, samples, threshold_upper)

    _assert_equal_vec(self, should_be_samples, samples)
    _assert_equal_vec(self, should_be_resamples, resamples)
    _assert_equal_vec(self, log_uniform_weights, should_be_uniform_weights)
    _assert_equal_vec(self, should_be_log_weights, log_weights)

if __name__ == '__main__':
  absltest.main()
