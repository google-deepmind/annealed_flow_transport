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

"""Code for normalizing flows.

For a review of normalizing flows see: https://arxiv.org/abs/1912.02762

The abstract base class ConfigurableFlow demonstrates our minimal interface.

Although the standard change of variables formula requires that
normalizing flows are invertible, none of the algorithms in train.py
require evaluating that inverse explicitly so inverses are not implemented.
"""

import abc
from typing import Callable, List, Tuple

import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Array = tp.Array
ConfigDict = tp.ConfigDict


class ConfigurableFlow(hk.Module, abc.ABC):
  """Abstract base clase for configurable normalizing flows.

  This is the interface expected by all flow based algorithms called in train.py
  """

  def __init__(self, config: ConfigDict):
    super().__init__()
    self._check_configuration(config)
    self._config = config

  def _check_input(self, x: Array) -> Array:
    chex.assert_rank(x, 1)

  def _check_outputs(self, x: Array, transformed_x: Array,
                     log_abs_det_jac: Array) -> Array:
    chex.assert_rank(x, 1)
    chex.assert_equal_shape([x, transformed_x])
    chex.assert_shape(log_abs_det_jac, ())

  def _check_members_types(self, config: ConfigDict, expected_members_types):
    for elem, elem_type in expected_members_types:
      if elem not in config:
        raise ValueError('Flow config element not found: ', elem)
      if not isinstance(config[elem], elem_type):
        msg = 'Flow config element '+elem+' is not of type '+str(elem_type)
        raise TypeError(msg)

  def __call__(self, x: Array) -> Tuple[Array, Array]:
    """Call transform_and_log abs_det_jac with automatic shape checking.

    This calls transform_and_log_abs_det_jac which needs to be implemented
    in derived classes.

    Args:
      x: Array size (num_dim,) containing input to flow.
    Returns:
      Array size (num_dim,) containing output and Scalar log abs det Jacobian.
    """
    self._check_input(x)
    output, log_abs_det_jac = self.transform_and_log_abs_det_jac(x)
    self._check_outputs(x, output, log_abs_det_jac)
    return output, log_abs_det_jac

  @abc.abstractmethod
  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    """Transform x through the flow and compute log abs determinant of Jacobian.

    Args:
      x: (num_dim,) input to the flow.
    Returns:
      Array size (num_dim,) containing output and Scalar log abs det Jacobian.
    """

  @abc.abstractmethod
  def _check_configuration(self, config: ConfigDict):
    """Check the configuration includes the necessary fields.

    Will typically raise Assertion like errors.

    Args:
      config: A ConfigDict include the fields required by the flow.
    """


class DiagonalAffine(ConfigurableFlow):
  """An affine transformation with a positive diagonal matrix."""

  def _check_configuration(self, unused_config: ConfigDict):
    pass

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    num_elem = x.shape[0]
    unconst_diag_init = hk.initializers.Constant(jnp.zeros((num_elem,)))
    bias_init = hk.initializers.Constant(jnp.zeros((num_elem,)))
    unconst_diag = hk.get_parameter('unconst_diag',
                                    shape=[num_elem],
                                    dtype=x.dtype,
                                    init=unconst_diag_init)
    bias = hk.get_parameter('bias',
                            shape=[num_elem],
                            dtype=x.dtype,
                            init=bias_init)
    output = jnp.exp(unconst_diag)*x + bias
    log_abs_det = jnp.sum(unconst_diag)
    return output, log_abs_det


def rational_quadratic_spline(x: Array,
                              bin_positions: Array,
                              bin_heights: Array,
                              derivatives: Array) -> Tuple[Array, Array]:
  """Compute a rational quadratic spline.

  See https://arxiv.org/abs/1906.04032

  Args:
    x: A single real number.
    bin_positions: A sorted array of bin positions of length num_bins+1.
    bin_heights: An array of bin heights of length num_bins+1.
    derivatives: An array of derivatives at bin positions of length num_bins+1.

  Returns:
    Value of the rational quadratic spline at x.
    Derivative with respect to x of rational quadratic spline at x.
  """

  bin_index = jnp.searchsorted(bin_positions, x)
  array_index = bin_index % len(bin_positions)
  lower_x = bin_positions[array_index-1]
  upper_x = bin_positions[array_index]
  lower_y = bin_heights[array_index-1]
  upper_y = bin_heights[array_index]
  lower_deriv = derivatives[array_index-1]
  upper_deriv = derivatives[array_index]
  delta_x = upper_x - lower_x
  delta_y = upper_y - lower_y
  slope = delta_y / delta_x
  alpha = (x - lower_x)/delta_x
  alpha_squared = jnp.square(alpha)
  beta = alpha * (1.-alpha)
  gamma = jnp.square(1.-alpha)
  epsilon = upper_deriv+lower_deriv -2. *slope
  numerator_quadratic = delta_y * (slope*alpha_squared + lower_deriv*beta)
  denominator_quadratic = slope + epsilon*beta
  interp_x = lower_y + numerator_quadratic/denominator_quadratic

  # now compute derivative
  numerator_deriv = jnp.square(slope) * (
      upper_deriv * alpha_squared + 2. * slope * beta + lower_deriv * gamma)
  sqrt_denominator_deriv = slope + epsilon*beta
  denominator_deriv = jnp.square(sqrt_denominator_deriv)
  deriv = numerator_deriv / denominator_deriv
  return interp_x, deriv


def identity_padded_rational_quadratic_spline(
    x: Array, bin_positions: Array, bin_heights: Array,
    derivatives: Array) -> Tuple[Array, Array]:
  """An identity padded rational quadratic spline.

  Args:
    x: the value to evaluate the spline at.
    bin_positions: sorted values of bin x positions of length num_bins+1.
    bin_heights: absolute height of bin of length num_bins-1.
    derivatives: derivatives at internal bin edge of length num_bins-1.

  Returns:
    The value of the spline at x.
    The derivative with respect to x of the spline at x.
  """
  lower_limit = bin_positions[0]
  upper_limit = bin_positions[-1]
  bin_height_sequence = (jnp.atleast_1d(jnp.array(lower_limit)),
                         bin_heights,
                         jnp.atleast_1d(jnp.array(upper_limit)))
  full_bin_heights = jnp.concatenate(bin_height_sequence)
  derivative_sequence = (jnp.ones((1,)),
                         derivatives,
                         jnp.ones((1,)))
  full_derivatives = jnp.concatenate(derivative_sequence)
  in_range = jnp.logical_and(jnp.greater(x, lower_limit),
                             jnp.less(x, upper_limit))
  multiplier = in_range*1.
  multiplier_complement = jnp.logical_not(in_range)*1.
  spline_val, spline_deriv = rational_quadratic_spline(x,
                                                       bin_positions,
                                                       full_bin_heights,
                                                       full_derivatives)
  identity_val = x
  identity_deriv = 1.
  val = spline_val*multiplier + multiplier_complement*identity_val
  deriv = spline_deriv*multiplier + multiplier_complement*identity_deriv
  return val, deriv


class AutoregressiveMLP(hk.Module):
  """An MLP which is constrained to have autoregressive dependency."""

  def __init__(self,
               num_hiddens_per_input_dim: List[int],
               include_self_links: bool,
               non_linearity,
               zero_final: bool,
               bias_last: bool,
               name=None):
    super().__init__(name=name)
    self._num_hiddens_per_input_dim = num_hiddens_per_input_dim
    self._include_self_links = include_self_links
    self._non_linearity = non_linearity
    self._zero_final = zero_final
    self._bias_last = bias_last

  def __call__(self, x: Array) -> Array:
    input_dim = x.shape[0]
    hidden_representation = jnp.atleast_2d(x).T
    prev_hid_per_dim = 1
    num_hidden_layers = len(self._num_hiddens_per_input_dim)
    final_index = num_hidden_layers-1

    for layer_index in range(num_hidden_layers):
      is_last_layer = (final_index == layer_index)
      hid_per_dim = self._num_hiddens_per_input_dim[layer_index]
      name_stub = '_'+str(layer_index)
      layer_shape = (input_dim,
                     prev_hid_per_dim,
                     input_dim,
                     hid_per_dim)
      in_degree = prev_hid_per_dim * input_dim
      if is_last_layer and self._zero_final:
        w_init = jnp.zeros
      else:
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(in_degree))
      bias_init = hk.initializers.Constant(jnp.zeros((input_dim, hid_per_dim,)))
      weights = hk.get_parameter(name='weights'+name_stub,
                                 shape=layer_shape,
                                 dtype=x.dtype,
                                 init=w_init)
      if is_last_layer and not self._bias_last:
        biases = jnp.zeros((input_dim, hid_per_dim,))
      else:
        biases = hk.get_parameter(name='biases'+name_stub,
                                  shape=(input_dim, hid_per_dim),
                                  dtype=x.dtype,
                                  init=bias_init)
      if not(self._include_self_links) and is_last_layer:
        k = -1
      else:
        k = 0
      mask = jnp.tril(jnp.ones((input_dim, input_dim)),
                      k=k)
      masked_weights = mask[:, None, :, None] * weights
      new_hidden_representation = jnp.einsum('ijkl,ij->kl',
                                             masked_weights,
                                             hidden_representation) + biases
      prev_hid_per_dim = hid_per_dim
      if not is_last_layer:
        hidden_representation = self._non_linearity(new_hidden_representation)
      else:
        hidden_representation = new_hidden_representation

    return hidden_representation


class InverseAutogressiveFlow(object):
  """A generic inverse autoregressive flow.

  See https://arxiv.org/abs/1606.04934

  Takes two functions as input.
  1) autoregressive_func takes array of (num_dim,)
  and returns array (num_dim, num_features)
  it is autoregressive in the sense that the output[i, :]
  depends only on the input[:i]. This is not checked.

  2) transform_func takes array of (num_dim, num_features) and
  an array of (num_dim,) and returns output of shape (num_dim,)
  and a single log_det_jacobian value. The represents the transformation
  acting on the inputs with given parameters.
  """

  def __init__(self,
               autoregressive_func: Callable[[Array], Array],
               transform_func: Callable[[Array, Array], Tuple[Array, Array]]):

    self._autoregressive_func = autoregressive_func
    self._transform_func = transform_func

  def __call__(self, x: Array) -> Tuple[Array, Array]:
    """x is of shape (num_dim,)."""
    transform_features = self._autoregressive_func(x)
    output, log_abs_det = self._transform_func(transform_features, x)
    return output, log_abs_det


class SplineInverseAutoregressiveFlow(ConfigurableFlow):
  """An inverse autoregressive flow with spline transformer.

  config must contain the following fields:
    num_spline_bins: Number of bins for rational quadratic spline.
    intermediate_hids_per_dim: See AutoregresiveMLP.
    num_layers: Number of layers for AutoregressiveMLP.
    identity_init: Whether to initalize the flow to the identity.
    bias_last: Whether to include biases on the last later of AutoregressiveMLP
    lower_lim: Lower limit of active region for rational quadratic spline.
    upper_lim: Upper limit of active region for rational quadratic spline.
    min_bin_size: Minimum bin size for rational quadratic spline.
    min_derivative: Minimum derivative for rational quadratic spline.
  """

  def __init__(self,
               config: ConfigDict):
    super().__init__(config)
    self._num_spline_bins = config.num_spline_bins
    num_spline_parameters = 3 * config.num_spline_bins - 1
    num_hids_per_input_dim = [config.intermediate_hids_per_dim
                             ] * config.num_layers + [
                                 num_spline_parameters
                             ]
    self._autoregressive_mlp = AutoregressiveMLP(
        num_hids_per_input_dim,
        include_self_links=False,
        non_linearity=jax.nn.leaky_relu,
        zero_final=config.identity_init,
        bias_last=config.bias_last)
    self._lower_lim = config.lower_lim
    self._upper_lim = config.upper_lim
    self._min_bin_size = config.min_bin_size
    self._min_derivative = config.min_derivative

  def _check_configuration(self, config: ConfigDict):
    expected_members_types = [
        ('num_spline_bins', int),
        ('intermediate_hids_per_dim', int),
        ('num_layers', int),
        ('identity_init', bool),
        ('bias_last', bool),
        ('lower_lim', float),
        ('upper_lim', float),
        ('min_bin_size', float),
        ('min_derivative', float)
    ]

    self._check_members_types(config, expected_members_types)

  def _unpack_spline_params(self, raw_param_vec) -> Tuple[Array, Array, Array]:
    unconst_bin_size_x = raw_param_vec[:self._num_spline_bins]
    unconst_bin_size_y = raw_param_vec[self._num_spline_bins:2 *
                                       self._num_spline_bins]
    unconst_derivs = raw_param_vec[2 * self._num_spline_bins:(
        3 * self._num_spline_bins - 1)]
    return unconst_bin_size_x, unconst_bin_size_y, unconst_derivs

  def _transform_raw_to_spline_params(
      self, raw_param_vec: Array) -> Tuple[Array, Array, Array]:
    unconst_bin_size_x, unconst_bin_size_y, unconst_derivs = self._unpack_spline_params(
        raw_param_vec)

    def normalize_bin_sizes(unconst_bin_sizes: Array) -> Array:
      bin_range = self._upper_lim - self._lower_lim
      reduced_bin_range = (
          bin_range - self._num_spline_bins * self._min_bin_size)
      return jax.nn.softmax(
          unconst_bin_sizes) * reduced_bin_range + self._min_bin_size

    bin_size_x = normalize_bin_sizes(unconst_bin_size_x)
    bin_size_y = normalize_bin_sizes(unconst_bin_size_y)

    # get the x bin positions.
    array_sequence = (jnp.ones((1,))*self._lower_lim, bin_size_x)
    x_bin_pos = jnp.cumsum(jnp.concatenate(array_sequence))

    # get the y bin positions, ignoring redundant terms.
    stripped_y_bin_pos = self._lower_lim + jnp.cumsum(bin_size_y[:-1])

    def forward_positive_transform(unconst_value: Array,
                                   min_value: Array) -> Array:
      return jax.nn.softplus(unconst_value) + min_value

    def inverse_positive_transform(const_value: Array,
                                   min_value: Array) -> Array:
      return jnp.log(jnp.expm1(const_value-min_value))

    inverted_one = inverse_positive_transform(1., self._min_derivative)
    derivatives = forward_positive_transform(unconst_derivs + inverted_one,
                                             self._min_derivative)
    return x_bin_pos, stripped_y_bin_pos, derivatives

  def _get_spline_values(self,
                         raw_parameters: Array,
                         x: Array) -> Tuple[Array, Array]:
    bat_get_parameters = jax.vmap(self._transform_raw_to_spline_params)
    bat_x_bin_pos, bat_stripped_y, bat_derivatives = bat_get_parameters(
        raw_parameters)
    # Vectorize spline over data and parameters.
    bat_get_spline_vals = jax.vmap(identity_padded_rational_quadratic_spline,
                                   in_axes=[0, 0, 0, 0])
    spline_vals, derivs = bat_get_spline_vals(x, bat_x_bin_pos, bat_stripped_y,
                                              bat_derivatives)
    log_abs_det = jnp.sum(jnp.log(jnp.abs(derivs)))
    return spline_vals, log_abs_det

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    iaf = InverseAutogressiveFlow(self._autoregressive_mlp,
                                  self._get_spline_values)
    return iaf(x)


class AffineInverseAutoregressiveFlow(ConfigurableFlow):
  """An inverse autoregressive flow with affine transformer.

  config must contain the following fields:
    intermediate_hids_per_dim: See AutoregresiveMLP.
    num_layers: Number of layers for AutoregressiveMLP.
    identity_init: Whether to initalize the flow to the identity.
    bias_last: Whether to include biases on the last later of AutoregressiveMLP
  """

  def __init__(self,
               config: ConfigDict):
    super().__init__(config)
    num_affine_params = 2
    num_hids_per_input_dim = [config.intermediate_hids_per_dim
                             ] * config.num_layers + [num_affine_params]
    self._autoregressive_mlp = AutoregressiveMLP(
        num_hids_per_input_dim,
        include_self_links=False,
        non_linearity=jax.nn.leaky_relu,
        zero_final=config.identity_init,
        bias_last=config.bias_last)

  def _check_configuration(self, config: ConfigDict):
    expected_members_types = [('intermediate_hids_per_dim', int),
                              ('num_layers', int),
                              ('identity_init', bool),
                              ('bias_last', bool)
                              ]

    self._check_members_types(config, expected_members_types)

  def _get_affine_transformation(self,
                                 raw_parameters: Array,
                                 x: Array) -> Tuple[Array, Array]:
    shifts = raw_parameters[:, 0]
    scales = raw_parameters[:, 1] + jnp.ones_like(raw_parameters[:, 1])
    log_abs_det = jnp.sum(jnp.log(jnp.abs(scales)))
    output = x * scales + shifts
    return output, log_abs_det

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    iaf = InverseAutogressiveFlow(self._autoregressive_mlp,
                                  self._get_affine_transformation)
    return iaf(x)


def affine_transformation(params: Array,
                          x: Array) -> Tuple[Array, Array]:
  shift = params[0]
  # Assuming params start as zero adding 1 to scale gives identity transform.
  scale = params[1] + 1.
  output = x * scale + shift
  return output, jnp.log(jnp.abs(scale))


class RationalQuadraticSpline(ConfigurableFlow):
  """A learnt monotonic rational quadratic spline with identity padding.

  Each input dimension is operated on by a separate spline.

  The spline is initialized to the identity.

  config must contain the following fields:
    num_bins: Number of bins for rational quadratic spline.
    lower_lim: Lower limit of active region for rational quadratic spline.
    upper_lim: Upper limit of active region for rational quadratic spline.
    min_bin_size: Minimum bin size for rational quadratic spline.
    min_derivative: Minimum derivative for rational quadratic spline.
  """

  def __init__(self,
               config: ConfigDict):
    super().__init__(config)
    self._num_bins = config.num_bins
    self._lower_lim = config.lower_lim
    self._upper_lim = config.upper_lim
    self._min_bin_size = config.min_bin_size
    self._min_derivative = config.min_derivative

  def _check_configuration(self, config: ConfigDict):
    expected_members_types = [
        ('num_bins', int),
        ('lower_lim', float),
        ('upper_lim', float),
        ('min_bin_size', float),
        ('min_derivative', float)
    ]

    self._check_members_types(config, expected_members_types)

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    """Apply the spline transformation.

    Args:
      x: (num_dim,) DeviceArray representing flow input.

    Returns:
      output: (num_dim,) transformed sample through flow.
      log_prob_out: new Scalar representing log_probability of output.
    """

    num_dim = x.shape[0]
    bin_parameter_shape = (num_dim, self._num_bins)

    # Setup the bin position and height parameters.
    bin_init = hk.initializers.Constant(jnp.ones(bin_parameter_shape))
    unconst_bin_size_x = hk.get_parameter(
        'unconst_bin_size_x',
        shape=bin_parameter_shape,
        dtype=x.dtype,
        init=bin_init)
    unconst_bin_size_y = hk.get_parameter(
        'unconst_bin_size_y',
        shape=bin_parameter_shape,
        dtype=x.dtype,
        init=bin_init)

    def normalize_bin_sizes(unconst_bin_sizes):
      bin_range = self._upper_lim - self._lower_lim
      reduced_bin_range = (bin_range - self._num_bins * self._min_bin_size)
      return jax.nn.softmax(
          unconst_bin_sizes) * reduced_bin_range + self._min_bin_size

    batched_normalize = jax.vmap(normalize_bin_sizes)
    bin_size_x = batched_normalize(unconst_bin_size_x)
    bin_size_y = batched_normalize(unconst_bin_size_y)
    array_sequence = (jnp.ones((num_dim, 1)) * self._lower_lim, bin_size_x)
    bin_positions = jnp.cumsum(jnp.concatenate(array_sequence, axis=1), axis=1)
    # Don't include the redundant bin heights.
    stripped_bin_heights = self._lower_lim + jnp.cumsum(
        bin_size_y[:, :-1], axis=1)

    # Setup the derivative parameters.

    def forward_positive_transform(unconst_value, min_value):
      return jax.nn.softplus(unconst_value) + min_value

    def inverse_positive_transform(const_value, min_value):
      return jnp.log(jnp.expm1(const_value - min_value))

    deriv_parameter_shape = (num_dim, self._num_bins - 1)
    inverted_one = inverse_positive_transform(1., self._min_derivative)
    deriv_init = hk.initializers.Constant(
        jnp.ones(deriv_parameter_shape) * inverted_one)
    unconst_deriv = hk.get_parameter(
        'unconst_deriv',
        shape=deriv_parameter_shape,
        dtype=x.dtype,
        init=deriv_init)
    batched_positive_transform = jax.vmap(
        forward_positive_transform, in_axes=[0, None])
    deriv = batched_positive_transform(unconst_deriv, self._min_derivative)

    # Setup batching then apply the spline.
    batch_padded_rq_spline = jax.vmap(
        identity_padded_rational_quadratic_spline, in_axes=[0, 0, 0, 0])
    output, jac_terms = batch_padded_rq_spline(x, bin_positions,
                                               stripped_bin_heights, deriv)
    log_abs_det_jac = jnp.sum(jnp.log(jac_terms))
    return output, log_abs_det_jac


def expand_periodic_dim(x: Array, num_extra_vals: int):
  if num_extra_vals == 0:
    return x
  first = x[-num_extra_vals:, :]
  last = x[:num_extra_vals, :]
  return jnp.vstack([first, x, last])


def pad_periodic_2d(x: Array, kernel_shape) -> Array:
  """Pad x to be have the required extra terms at the edges."""
  assert len(kernel_shape) == 2
  chex.assert_rank(x, 2)
  # this code is unbatched
  # we require that kernel shape has odd rows/cols.
  is_even = False
  for elem in kernel_shape:
    is_even = is_even or (elem % 2 == 0)
  if is_even:
    raise ValueError('kernel_shape is assumed to have odd rows and cols')
  # calculate num extra rows/cols each side.
  num_extra_row = (kernel_shape[0] - 1) // 2
  num_extra_col = (kernel_shape[1] -1) // 2
  row_expanded_x = expand_periodic_dim(x,
                                       num_extra_row)
  col_expanded_x = expand_periodic_dim(row_expanded_x.T,
                                       num_extra_col).T
  return col_expanded_x


def batch_pad_periodic_2d(x: Array, kernel_shape) -> Array:
  assert len(kernel_shape) == 2
  chex.assert_rank(x, 4)
  batch_func = jax.vmap(pad_periodic_2d, in_axes=(0, None))
  batch_channel_func = jax.vmap(batch_func, in_axes=(3, None), out_axes=3)
  return batch_channel_func(x, kernel_shape)


class Conv2DTorus(hk.Conv2D):
  """Convolution in 2D with periodic boundary conditions.

  Strides are ignored and this is not checked.
  kernel_shapes is a tuple (a, b) where a and b are odd positive integers.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, padding='VALID', **kwargs)

  def __call__(self, x: Array) -> Array:
    padded_x = batch_pad_periodic_2d(x, self.kernel_shape)
    return super().__call__(padded_x)


class FullyConvolutionalNetwork(hk.Module):
  """A fully convolutional network with ResNet middle layers."""

  def __init__(self,
               num_middle_channels: int = 5,
               num_middle_layers: int = 2,
               num_final_channels: int = 2,
               kernel_shape: Tuple[int] = (3, 3),
               zero_final: bool = True,
               is_torus: bool = False):  # pytype: disable=annotation-type-mismatch
    super().__init__()
    self._num_middle_channels = num_middle_channels
    self._num_middle_layers = num_middle_layers
    self._num_final_channels = num_final_channels
    self._kernel_shape = kernel_shape
    self._zero_final = zero_final
    self._is_torus = is_torus

  def __call__(self,
               x: Array):
    """Call the residual network on x.

    Args:
      x: is of shape (length_a, length_b)
    Returns:
      Array of shape (length_a, length_b, num_channels[-1])
    """
    chex.assert_rank(x, 2)
    length_a, length_b = jnp.shape(x)
    non_linearity = jax.nn.relu
    if self._is_torus:
      conv_two_d = Conv2DTorus
    else:
      conv_two_d = hk.Conv2D
    # Cast to batch size of one and one channel in last index.
    representation = x[None, :, :, None]

    for middle_layer_index in range(self._num_middle_layers):
      if middle_layer_index == 0:
        representation = conv_two_d(
            output_channels=self._num_middle_channels,
            stride=1,
            kernel_shape=self._kernel_shape,
            with_bias=True)(representation)
        representation = non_linearity(representation)
      else:
        conv_result = conv_two_d(
            output_channels=self._num_middle_channels,
            stride=1,
            kernel_shape=self._kernel_shape,
            with_bias=True)(representation)
        representation = representation + non_linearity(conv_result)
    if self._zero_final:
      representation = conv_two_d(
          output_channels=self._num_final_channels,
          stride=1,
          kernel_shape=self._kernel_shape,
          with_bias=True,
          w_init=jnp.zeros,
          b_init=jnp.zeros)(representation)
    else:
      representation = conv_two_d(
          output_channels=self._num_final_channels,
          stride=1,
          kernel_shape=self._kernel_shape,
          with_bias=True)(representation)
    chex.assert_shape(representation,
                      [1, length_a, length_b, self._num_final_channels])
    # Remove extraneous batch index of size 1.
    representation = representation[0, :, :, :]
    return representation


class CouplingLayer(object):
  """A generic coupling layer.

  Takes the following functions as inputs.
  1) A conditioner network mapping from event_shape->event_shape + (num_params,)
  2) Mask of shape event_shape.
  3) transformer A map from event_shape -> event_shape that acts elementwise on
  the terms to give a diagonal Jacobian expressed as shape event_shape and in
  abs-log space.
  It is parameterised by parameters of shape params_shape.

  """

  def __init__(self, conditioner_network: Callable[[Array], Array], mask: Array,
               transformer: Callable[[Array, Array], Tuple[Array, Array]]):
    self._conditioner_network = conditioner_network
    self._mask = mask
    self._transformer = transformer

  def __call__(self, x):
    """Transform x with coupling layer.

    Args:
      x: event_shape Array.
    Returns:
      output_x: event_shape Array corresponding to the output.
      log_abs_det: scalar corresponding to the log abs det Jacobian.
    """
    mask_complement = 1. - self._mask
    masked_x = x * self._mask
    chex.assert_equal_shape([masked_x, x])
    transformer_params = self._conditioner_network(masked_x)
    transformed_x, log_abs_dets = self._transformer(transformer_params, x)
    output_x = masked_x + mask_complement * transformed_x
    chex.assert_equal_shape([transformed_x,
                             output_x,
                             x,
                             log_abs_dets])
    log_abs_det = jnp.sum(log_abs_dets * mask_complement)
    return output_x, log_abs_det


class ConvAffineCoupling(CouplingLayer):
  """A convolutional affine coupling layer."""

  def __init__(self,
               mask: Array,
               conv_num_middle_channels: int = 5,
               conv_num_middle_layers: int = 2,
               conv_kernel_shape: Tuple[int] = (3, 3),
               identity_init: bool = True,
               is_torus: bool = False):  # pytype: disable=annotation-type-mismatch
    conv_net = FullyConvolutionalNetwork(
        num_middle_channels=conv_num_middle_channels,
        num_middle_layers=conv_num_middle_layers,
        num_final_channels=2,
        kernel_shape=conv_kernel_shape,
        zero_final=identity_init,
        is_torus=is_torus)
    vectorized_affine = jnp.vectorize(affine_transformation,
                                      signature='(k),()->(),()')

    super().__init__(conv_net,
                     mask,
                     vectorized_affine)


def get_checkerboard_mask(overall_shape: Tuple[int, int],
                          period: int):
  range_a = jnp.arange(overall_shape[0])
  range_b = jnp.arange(overall_shape[1])
  def modulo_func(index_a, index_b):
    return jnp.mod(index_a+index_b+period, 2)
  func = lambda y: jax.vmap(modulo_func, in_axes=[0, None])(range_a, y)
  vals = func(range_b)
  chex.assert_shape(vals, overall_shape)
  return vals


class ConvAffineCouplingStack(ConfigurableFlow):
  """A stack of convolutional affine coupling layers."""

  def __init__(self, config: ConfigDict):
    super().__init__(config)
    num_elem = config.num_elem
    num_grid_per_dim = int(np.sqrt(num_elem))
    assert num_grid_per_dim * num_grid_per_dim == num_elem
    self._true_shape = (num_grid_per_dim, num_grid_per_dim)
    self._coupling_layers = []
    for index in range(self._config.num_coupling_layers):
      mask = get_checkerboard_mask(self._true_shape, index)
      coupling_layer = ConvAffineCoupling(
          mask,
          conv_kernel_shape=self._config.conv_kernel_shape,
          conv_num_middle_layers=self._config.conv_num_middle_layers,
          conv_num_middle_channels=self._config.conv_num_middle_channels,
          is_torus=self._config.is_torus,
          identity_init=self._config.identity_init
      )
      self._coupling_layers.append(coupling_layer)

  def _check_configuration(self, config: ConfigDict):
    expected_members_types = [
        ('conv_kernel_shape', list),
        ('conv_num_middle_layers', int),
        ('conv_num_middle_channels', int),
        ('is_torus', bool),
        ('identity_init', bool)
    ]

    self._check_members_types(config, expected_members_types)

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    reshaped_x = jnp.reshape(x, self._true_shape)
    transformed_x = reshaped_x
    log_abs_det = 0.
    for index in range(self._config.num_coupling_layers):
      coupling_layer = self._coupling_layers[index]
      transformed_x, log_det_increment = coupling_layer(transformed_x)
      chex.assert_equal_shape([transformed_x, reshaped_x])
      log_abs_det += log_det_increment
    restored_x = jnp.reshape(transformed_x, x.shape)
    return restored_x, log_abs_det


class ComposedFlows(ConfigurableFlow):
  """Class to compose flows based on a list of configs.

  config should contain flow_configs a list of flow configs to compose.
  """

  def __init__(self, config: ConfigDict):
    super().__init__(config)
    self._flows = []
    for flow_config in self._config.flow_configs:
      base_flow_class = globals()[flow_config.type]
      flow = base_flow_class(flow_config)
      self._flows.append(flow)

  def _check_configuration(self, config: ConfigDict):
    expected_members_types = [
        ('flow_configs', list),
    ]

    self._check_members_types(config, expected_members_types)

  def transform_and_log_abs_det_jac(self, x: Array) -> Tuple[Array, Array]:
    log_abs_det = 0.
    progress = x
    for flow in self._flows:
      progress, log_abs_det_increment = flow(progress)
      log_abs_det += log_abs_det_increment
    return progress, log_abs_det
