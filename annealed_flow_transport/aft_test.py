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

"""Tests for annealed_flow_transport.aft."""

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport import aft
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import optax


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class AftTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    num_dim = 1
    num_samples = 4
    self._train_samples = jnp.zeros((num_samples, num_dim))
    self._train_log_weights = -jnp.log(num_samples)*jnp.ones((num_samples,))
    self._validation_samples = self._train_samples
    self._validation_log_weights = self._train_log_weights
    self._true_target = jnp.ones((num_dim,))
    self._initial_mean = jnp.zeros((num_dim,))
    self._opt = optax.adam(1e-2)
    self._opt_init_state = self._opt.init(self._initial_mean)
    self.dummy_free_energy_and_grad = jax.value_and_grad(self.dummy_free_energy)
    self._dummy_outer_step = 0
    self._iterations = 500

  def dummy_free_energy(self, mean, samples, log_weights, unused_step):
    integrands = jnp.square(samples + mean - self._true_target[None, :])[:, 0]
    return jnp.sum(jax.nn.softmax(log_weights) * integrands)

  def test_early_stopping(self):
    best_mean_greedy, unused_opt_values = aft.optimize_free_energy(
        opt_update=self._opt.update,
        opt_init_state=self._opt_init_state,
        flow_init_params=self._initial_mean,
        free_energy_and_grad=self.dummy_free_energy_and_grad,
        free_energy_eval=self.dummy_free_energy,
        train_samples=self._train_samples,
        train_log_weights=self._train_log_weights,
        validation_samples=self._validation_samples,
        validation_log_weights=self._validation_log_weights,
        outer_step=self._dummy_outer_step,
        opt_iters=self._iterations,
        stopping_criterion='greedy_time')
    best_mean_time, opt_values = aft.optimize_free_energy(
        opt_update=self._opt.update,
        opt_init_state=self._opt_init_state,
        flow_init_params=self._initial_mean,
        free_energy_and_grad=self.dummy_free_energy_and_grad,
        free_energy_eval=self.dummy_free_energy,
        train_samples=self._train_samples,
        train_log_weights=self._train_log_weights,
        validation_samples=self._validation_samples,
        validation_log_weights=self._validation_log_weights,
        outer_step=self._dummy_outer_step,
        opt_iters=self._iterations,
        stopping_criterion='time')
    _assert_equal_vec(self, best_mean_greedy, self._true_target, atol=1e-5)
    _assert_equal_vec(self, best_mean_time, self._true_target, atol=1e-5)

    min_train_vfe = jnp.min(opt_values.train_vfes)
    min_validation_vfe = jnp.min(opt_values.validation_vfes)
    true_minimium = 0.

    _assert_equal_vec(self, min_train_vfe, true_minimium, atol=1e-5)
    _assert_equal_vec(self, min_validation_vfe, true_minimium, atol=1e-5)

  def test_opt_step(self):
    initial_vfes = tp.VfesTuple(jnp.zeros(self._iterations),
                                jnp.zeros(self._iterations))
    initial_loop_state = aft.OptimizationLoopState(
        self._opt_init_state,
        self._initial_mean,
        inner_step=0,
        opt_vfes=initial_vfes,
        best_params=self._initial_mean,
        best_validation_vfe=jnp.inf,
        best_index=-1)
    loop_state_b = aft.flow_estimate_step(
        loop_state=initial_loop_state,
        free_energy_and_grad=self.dummy_free_energy_and_grad,
        train_samples=self._train_samples,
        train_log_weights=self._train_log_weights,
        outer_step=self._dummy_outer_step,
        validation_samples=self._validation_samples,
        validation_log_weights=self._validation_log_weights,
        free_energy_eval=self.dummy_free_energy,
        opt_update=self._opt.update)
    self.assertEqual(loop_state_b.inner_step, 1)
    array_one = jnp.array(1.)
    _assert_equal_vec(self, loop_state_b.opt_vfes.train_vfes[0], array_one)
    _assert_equal_vec(self, loop_state_b.opt_vfes.validation_vfes[0], array_one)
    _assert_equal_vec(self, loop_state_b.best_params, self._initial_mean)
    _assert_equal_vec(self, loop_state_b.best_validation_vfe, array_one)
    self.assertEqual(loop_state_b.best_index, 0)

    loop_state_c = aft.flow_estimate_step(
        loop_state=loop_state_b,
        free_energy_and_grad=self.dummy_free_energy_and_grad,
        train_samples=self._train_samples,
        train_log_weights=self._train_log_weights,
        outer_step=self._dummy_outer_step,
        validation_samples=self._validation_samples,
        validation_log_weights=self._validation_log_weights,
        free_energy_eval=self.dummy_free_energy,
        opt_update=self._opt.update)
    self.assertEqual(loop_state_c.inner_step, 2)
    self.assertLess(loop_state_c.best_validation_vfe, array_one)
    self.assertEqual(loop_state_c.best_index, 1)


if __name__ == '__main__':
  absltest.main()
