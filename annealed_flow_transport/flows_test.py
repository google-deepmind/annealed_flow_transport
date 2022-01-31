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

"""Tests for annealed_flow_transport.flows."""

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport import flows
import haiku as hk
import jax
import jax.numpy as jnp

import ml_collections
ConfigDict = ml_collections.ConfigDict


def _assert_equal_vec(tester, v1, v2, **kwargs):
  tester.assertTrue(jnp.allclose(v1, v2, **kwargs))


class DiagonalAffineTest(parameterized.TestCase):

  def test_identity_init(self):
    # Config dict is unused here so pass None.
    def compute_flow(x_loc):
      diagonal_affine_flow = flows.DiagonalAffine(None)
      return diagonal_affine_flow(x_loc)
    x_in = jnp.array([1., 2., 3.])
    flow_func = hk.without_apply_rng(hk.transform(compute_flow))
    dummy_key = jax.random.PRNGKey(13)
    init_params = flow_func.init(dummy_key, x_in)
    x_out, log_det_abs_jac = flow_func.apply(init_params, x_in)
    _assert_equal_vec(self, x_out, x_in)
    _assert_equal_vec(self, log_det_abs_jac, 0.)

  def test_non_identity(self):
    # Config dict is unused here so pass None.
    def compute_flow(x_loc):
      diagonal_affine_flow = flows.DiagonalAffine(None)
      return diagonal_affine_flow(x_loc)
    x_in = jnp.array([1., 2., 3.])
    flow_func = hk.without_apply_rng(hk.transform(compute_flow))
    dummy_key = jax.random.PRNGKey(13)
    init_params = flow_func.init(dummy_key, x_in)
    shift = -0.3
    num_dim = 3
    new_params = jax.tree_map(lambda x: x+shift, init_params)
    x_out, log_det_abs_jac = flow_func.apply(new_params, x_in)
    validation_x_out = x_in * jnp.exp(shift) + shift
    _assert_equal_vec(self, validation_x_out, x_out)
    _assert_equal_vec(self, log_det_abs_jac, num_dim*shift)

    def short_func(x_loc):
      return flow_func.apply(new_params, x_loc)[0]

    numerical_jacobian = jax.jacobian(short_func)(x_in)
    numerical_log_abs_det = jnp.linalg.slogdet(numerical_jacobian)[1]
    _assert_equal_vec(self, log_det_abs_jac, numerical_log_abs_det)


class SplinesTest(parameterized.TestCase):

  def _get_non_identity_monotone_spline(self):
    bin_positions = jnp.linspace(-3., 3., 10)
    bin_heights = jax.nn.softplus(bin_positions)
    derivatives = jax.nn.sigmoid(bin_positions)
    def spline_func(x):
      return flows.rational_quadratic_spline(x,
                                             bin_positions,
                                             bin_heights,
                                             derivatives)
    return spline_func

  def test_identity(self):
    bin_positions = jnp.linspace(-3., 3., 10)
    bin_heights = bin_positions
    derivatives = jnp.ones_like(bin_heights)
    x = jnp.array(1.77)
    output, output_deriv = flows.rational_quadratic_spline(x,
                                                           bin_positions,
                                                           bin_heights,
                                                           derivatives)
    _assert_equal_vec(self, x, output)
    _assert_equal_vec(self, output_deriv, 1.)

  def test_jacobian(self):
    # Test jacobian against numerical value for non-identity transformation.

    x = jnp.array(1.77)
    spline_func = self._get_non_identity_monotone_spline()
    _, output_deriv = spline_func(x)
    curry = lambda input: spline_func(input)[0]
    grad_func = jax.grad(curry)
    grad_val = grad_func(x)
    _assert_equal_vec(self, grad_val, output_deriv)

  def test_monotonic(self):
    # Test function is monotonic and has positive deriviatives.
    x = jnp.linspace(-2.7, 2.7, 17)
    spline_func_bat = jax.vmap(self._get_non_identity_monotone_spline())
    spline_vals, spline_derivs = spline_func_bat(x)
    self.assertTrue((jnp.diff(spline_vals) > 0.).all())
    self.assertTrue((jnp.diff(spline_derivs) > 0.).all())


class AutoregressiveMLPTest(parameterized.TestCase):

  def _get_transformed(self, zero_init):
    def forward(x):
      mlp = flows.AutoregressiveMLP([3, 1],
                                    False,
                                    jax.nn.leaky_relu,
                                    zero_init,
                                    True)
      return mlp(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    return forward_fn

  def test_zero_init(self):
    forward_fn = self._get_transformed(True)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    output = forward_fn.apply(params, x)
    _assert_equal_vec(self, output, jnp.zeros_like(output))

  def test_autoregressive(self):
    forward_fn = self._get_transformed(False)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    curry = lambda u: forward_fn.apply(params, u)[:, 0]
    jacobian = jax.jacobian(curry)(x)
    lower_triangle = jnp.tril(jacobian, k=0)
    _assert_equal_vec(self, lower_triangle, jnp.zeros_like(lower_triangle))


class SplineInverseAutoregressiveFlowTest(parameterized.TestCase):

  def _get_config(self, identity_init):
    flow_config = ConfigDict()
    flow_config.num_spline_bins = 10
    flow_config.intermediate_hids_per_dim = 30
    flow_config.num_layers = 3
    flow_config.identity_init = identity_init
    flow_config.lower_lim = -4.
    flow_config.upper_lim = 4.
    flow_config.min_bin_size = 1e-4
    flow_config.min_derivative = 1e-4
    flow_config.bias_last = True
    return flow_config

  def _get_transformed(self, identity_init):
    def forward(x):
      config = self._get_config(identity_init)
      flow = flows.SplineInverseAutoregressiveFlow(config)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    return forward_fn

  def test_identity_init(self):
    forward_fn = self._get_transformed(True)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    output, log_det_jac = forward_fn.apply(params, x)
    _assert_equal_vec(self, output, x)
    _assert_equal_vec(self, 0., log_det_jac, atol=1e-6)

  def test_jacobian(self):
    # Compare the numerical Jacobian with computed value.
    forward_fn = self._get_transformed(False)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    curry_val = lambda x: forward_fn.apply(params, x)[0]
    curry_jac = lambda x: forward_fn.apply(params, x)[1]
    jac_func = jax.jacobian(curry_val)
    jac = jac_func(x)
    target_log_det_jac = jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
    test_log_det_jac = curry_jac(x)
    _assert_equal_vec(self, target_log_det_jac, test_log_det_jac)
    lower_triangle = jnp.tril(jac, k=-1)
    _assert_equal_vec(self, lower_triangle, jnp.zeros_like(lower_triangle))


class AffineInverseAutoregressiveFlowTest(parameterized.TestCase):

  def _get_config(self, identity_init):
    flow_config = ConfigDict()
    flow_config.intermediate_hids_per_dim = 30
    flow_config.num_layers = 3
    flow_config.identity_init = identity_init
    flow_config.bias_last = True
    return flow_config

  def _get_transformed(self, identity_init):
    def forward(x):
      config = self._get_config(identity_init)
      flow = flows.AffineInverseAutoregressiveFlow(config)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    return forward_fn

  def test_identity_init(self):
    forward_fn = self._get_transformed(True)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    output, log_det_jac = forward_fn.apply(params, x)
    _assert_equal_vec(self, output, x)
    _assert_equal_vec(self, 0., log_det_jac, atol=1e-6)

  def test_jacobian(self):
    # Compare the numerical Jacobian with computed value.
    forward_fn = self._get_transformed(False)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    curry_val = lambda x: forward_fn.apply(params, x)[0]
    curry_jac = lambda x: forward_fn.apply(params, x)[1]
    jac_func = jax.jacobian(curry_val)
    jac = jac_func(x)
    target_log_det_jac = jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
    test_log_det_jac = curry_jac(x)
    _assert_equal_vec(self, target_log_det_jac, test_log_det_jac)
    lower_triangle = jnp.tril(jac, k=-1)
    _assert_equal_vec(self, lower_triangle, jnp.zeros_like(lower_triangle))


class RationalQuadraticSplineFlowTest(parameterized.TestCase):
  """This just tests that the flow constructs and gives right shape.

  The math functions are separately tested by SplinesTest.
  """

  def _get_config(self):
    flow_config = ConfigDict()
    flow_config.num_bins = 5
    flow_config.lower_lim = -3.
    flow_config.upper_lim = 3.
    flow_config.min_bin_size = 1e-2
    flow_config.min_derivative = 1e-2
    return flow_config

  def test_shape(self):
    def forward(x):
      config = self._get_config()
      flow = flows.RationalQuadraticSpline(config)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    output, log_det_jac = forward_fn.apply(params, x)
    self.assertEqual(x.shape, output.shape)
    self.assertEqual(log_det_jac.shape, ())


class ComposedFlowsTest(parameterized.TestCase):

  def _get_individual_config(self, is_identity: bool):
    flow_config = ConfigDict()
    flow_config.type = 'SplineInverseAutoregressiveFlow'
    flow_config.num_spline_bins = 10
    flow_config.intermediate_hids_per_dim = 30
    flow_config.num_layers = 3
    flow_config.identity_init = is_identity
    flow_config.lower_lim = -4.
    flow_config.upper_lim = 4.
    flow_config.min_bin_size = 1e-4
    flow_config.min_derivative = 1e-4
    flow_config.bias_last = True
    return flow_config

  def _get_overall_config(self, is_identity: bool):
    flow_config = ConfigDict()
    # A flow based on two flows composed.
    flow_config.flow_configs = [self._get_individual_config(is_identity)] * 2
    return flow_config

  def _get_transformed(self, is_identity):
    def forward(x):
      config = self._get_overall_config(is_identity=is_identity)
      flow = flows.ComposedFlows(config)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    return forward_fn

  def test_identity(self):
    # Test that two identity flows composed gives an identity flow.
    forward_fn = self._get_transformed(is_identity=True)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    output, log_det_jac = forward_fn.apply(params, x)
    _assert_equal_vec(self, x, output, atol=1e-6)
    _assert_equal_vec(self, log_det_jac, 0., atol=1e-6)

  def test_jacobian(self):
    # Test the numerical Jacobian of the composition of two non-identity flows.
    forward_fn = self._get_transformed(is_identity=False)
    x = jnp.array([1., 2., 3.])
    key = jax.random.PRNGKey(13)
    params = forward_fn.init(key,
                             x)
    curry_val = lambda x: forward_fn.apply(params, x)[0]
    curry_jac = lambda x: forward_fn.apply(params, x)[1]
    jac_func = jax.jacobian(curry_val)
    jac = jac_func(x)
    target_log_det_jac = jnp.linalg.slogdet(jac)[1]
    test_log_det_jac = curry_jac(x)
    _assert_equal_vec(self, target_log_det_jac, test_log_det_jac, atol=1e-6)


class CheckerBoardMaskTest(parameterized.TestCase):

  def test_checkerboard(self):
    target_a = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    target_b = jnp.array([[1, 0, 1], [0, 1, 0]])
    test_a = flows.get_checkerboard_mask((3, 3), 0)
    test_b = flows.get_checkerboard_mask((2, 3), 1)
    _assert_equal_vec(self, target_a, test_a)
    _assert_equal_vec(self, target_b, test_b)


class TestFullyConvolutionalNetwork(parameterized.TestCase):

  def test_net(self):
    num_middle_channels = 3
    num_middle_layers = 2
    num_final_channels = 2
    image_shape = (7, 9)
    kernel_shape = (4, 3)
    def forward(x):
      net = flows.FullyConvolutionalNetwork(
          num_middle_channels=num_middle_channels,
          num_middle_layers=num_middle_layers,
          num_final_channels=num_final_channels,
          kernel_shape=kernel_shape,
          zero_final=True)
      return net(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(1)
    subkey, key = jax.random.split(key)
    random_input = jax.random.normal(subkey, image_shape)
    params = forward_fn.init(key, random_input)
    output = forward_fn.apply(params, random_input)
    self.assertEqual(output.shape, image_shape+(2,))
    _assert_equal_vec(self, output, jnp.zeros_like(output))

  def test_translation_symmetry(self):
    num_middle_channels = 3
    num_middle_layers = 2
    num_final_channels = 2
    image_shape = (7, 9)
    kernel_shape = (3, 3)
    def forward(x):
      net = flows.FullyConvolutionalNetwork(
          num_middle_channels=num_middle_channels,
          num_middle_layers=num_middle_layers,
          num_final_channels=num_final_channels,
          kernel_shape=kernel_shape,
          zero_final=False,
          is_torus=True)
      return net(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(1)
    subkey, key = jax.random.split(key)
    random_input = jax.random.normal(subkey, image_shape)
    params = forward_fn.init(key, random_input)
    output = forward_fn.apply(params, random_input)
    def roll_array(array_in):
      return jnp.roll(array_in,
                      shift=(2, 2),
                      axis=(0, 1))
    translated_output = forward_fn.apply(params,
                                         roll_array(random_input))
    _assert_equal_vec(self,
                      translated_output,
                      roll_array(output))


class TestConvAffineCoupling(parameterized.TestCase):

  def test_identity_init(self):
    image_shape = (3, 3)
    kernel_shape = (2, 2)
    num_middle_channels = 3
    num_middle_layers = 3
    mask = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    def forward(x):
      flow = flows.ConvAffineCoupling(
          mask=mask,
          conv_num_middle_channels=num_middle_channels,
          conv_num_middle_layers=num_middle_layers,
          conv_kernel_shape=kernel_shape,
          identity_init=True)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(1)
    subkey, key = jax.random.split(key)
    random_input = jax.random.normal(subkey, shape=image_shape)
    params = forward_fn.init(key, random_input)
    output, log_det_jac = forward_fn.apply(params, random_input)
    _assert_equal_vec(self, output, random_input)
    _assert_equal_vec(self, log_det_jac, 0.)

  def test_jacobian(self):
    num_middle_channels = 3
    num_middle_layers = 5
    image_shape = (3, 2)
    kernel_shape = (3, 3)
    mask = jnp.array([[1, 1], [1, 0], [0, 0]])
    def forward(x):
      flow = flows.ConvAffineCoupling(
          mask=mask,
          conv_num_middle_channels=num_middle_channels,
          conv_num_middle_layers=num_middle_layers,
          conv_kernel_shape=kernel_shape,
          identity_init=False)
      return flow(x)
    forward_fn = hk.without_apply_rng(hk.transform(forward))
    key = jax.random.PRNGKey(2)
    subkey, key = jax.random.split(key)
    random_input = jax.random.normal(subkey, shape=image_shape)
    params = forward_fn.init(key, random_input)
    apply = jax.jit(forward_fn.apply)
    curry_val = lambda x: apply(params, x)[0].reshape((6,))
    curry_jac = lambda x: apply(params, x)[1]
    jac_func = jax.jit(jax.jacobian(curry_val))
    jac = jac_func(random_input).reshape((6, 6))
    print('NVP Jacobian  \n', jac)
    target_log_det_jac = jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
    test_log_det_jac = curry_jac(random_input)
    _assert_equal_vec(self, target_log_det_jac, test_log_det_jac)
    upper_triangle = jnp.triu(jac, k=1)
    _assert_equal_vec(self, upper_triangle, jnp.zeros_like(upper_triangle))

if __name__ == '__main__':
  absltest.main()
