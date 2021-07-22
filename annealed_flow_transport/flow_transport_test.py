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
from annealed_flow_transport.flow_transport import GeometricAnnealingSchedule
from annealed_flow_transport.flow_transport import transport_free_energy_estimator
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class FlowTransportTest(parameterized.TestCase):

  def test_transport_free_energy_estimator(self):
    # parameters for various normal distribution.
    mean_a = 0.
    mean_b = -1.
    mean_c = 2.
    mean_d = 0.1
    var_a = 1.
    var_b = 4.
    var_c = 5.
    var_d = 1.

    num_samples = 10000
    dimension = 1
    key = jax.random.PRNGKey(1)

    # First we sample from normal distribution d
    # and then perform an importance correction to normal distribution a.
    samples = jax.random.normal(key, shape=(num_samples, dimension))+mean_d

    def log_importance_correction(samples):
      first_terms = norm.logpdf(
          samples, loc=mean_a, scale=jnp.sqrt(var_a)).flatten()
      second_terms = norm.logpdf(
          samples, loc=mean_d, scale=jnp.sqrt(var_d)).flatten()
      return first_terms-second_terms

    log_weights = log_importance_correction(samples)

    # this will change the normal distribution a to normal distribution b
    # because it is an affine transformation.
    def flow_apply(unused_params, samples):
      return 2.*samples-1., jnp.log(2)*jnp.ones((num_samples,))

    def analytic_gauss_kl(mean0, var0, mean1, var1):
      return 0.5 * (
          var0 / var1 + jnp.square(mean1 - mean0) / var1 - 1. + jnp.log(var1) -
          jnp.log(var0))

    def initial_density(x):
      return norm.logpdf(x, loc=mean_a, scale=jnp.sqrt(var_a)).flatten()

    def final_density(x):
      return norm.logpdf(x, loc=mean_c, scale=jnp.sqrt(var_c)).flatten()

    def step_density(step, x):
      if step == 0:
        return initial_density(x)
      if step == 1:
        return final_density(x)

    estimator_value = transport_free_energy_estimator(samples=samples,
                                                      log_weights=log_weights,
                                                      flow_apply=flow_apply,
                                                      flow_params=None,
                                                      log_density=step_density,
                                                      step=1)

    # the target KL is analytically tractable as it is between two Gaussians.
    # the target KL is between normal b and c.
    analytic_value = analytic_gauss_kl(mean0=mean_b,
                                       var0=var_b,
                                       mean1=mean_c,
                                       var1=var_c)

    _assert_equal_vec(self, estimator_value, analytic_value, atol=1e-2)

  def test_geometric_annealing_schedule(self):
    def initial_density(x):
      return norm.logpdf(x, loc=-1., scale=2.).flatten()

    def final_density(x):
      return norm.logpdf(x, loc=1.5, scale=3.).flatten()

    num_temps = 5.

    annealing_schedule = GeometricAnnealingSchedule(initial_density,
                                                    final_density,
                                                    num_temps)
    num_samples = 10000
    dimension = 1
    key = jax.random.PRNGKey(1)

    samples = jax.random.normal(key, shape=(num_samples, dimension))

    interpolated_densities_initial = annealing_schedule(0, samples)
    test_densities_initial = initial_density(samples)
    interpolated_densities_final = annealing_schedule(4, samples)
    test_densities_final = final_density(samples)

    _assert_equal_vec(self,
                      interpolated_densities_initial,
                      test_densities_initial)

    _assert_equal_vec(self,
                      interpolated_densities_final,
                      test_densities_final)

if __name__ == '__main__':
  absltest.main()
