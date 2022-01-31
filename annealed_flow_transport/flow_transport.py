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

"""Shared math functions for flow transport SMC algorithms."""
from typing import Tuple

from annealed_flow_transport import resampling
import annealed_flow_transport.aft_types as tp
import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = jnp.ndarray
FlowApply = tp.FlowApply
FlowParams = tp.FlowParams
LogDensityByStep = tp.LogDensityByStep
LogDensityNoStep = tp.LogDensityNoStep
MarkovKernelApply = tp.MarkovKernelApply
AcceptanceTuple = tp.AcceptanceTuple
RandomKey = tp.RandomKey
assert_equal_shape = chex.assert_equal_shape


class GeometricAnnealingSchedule(object):
  """Container computing a geometric annealing schedule between log densities."""

  def __init__(self,
               initial_log_density: LogDensityNoStep,
               final_log_density: LogDensityNoStep,
               num_temps: int):
    self._initial_log_density = initial_log_density
    self._final_log_density = final_log_density
    self._num_temps = num_temps

  def get_beta(self,
               step):
    final_step = self._num_temps-1
    beta = step / final_step
    return beta

  def __call__(self,
               step: int,
               samples: Array):
    log_densities_final = self._final_log_density(samples)
    log_densities_initial = self._initial_log_density(samples)
    beta = self.get_beta(step)
    interpolated_densities = (
        1. - beta) * log_densities_initial + beta * log_densities_final
    return interpolated_densities


def get_delta_no_flow(samples: Array,
                      log_density: LogDensityByStep,
                      step: int) -> Array:
  log_density_values_current = log_density(step, samples)
  log_density_values_previous = log_density(step-1, samples)
  assert_equal_shape([log_density_values_current, log_density_values_previous])
  deltas = log_density_values_previous - log_density_values_current
  return deltas


def get_delta(samples: Array,
              flow_apply: FlowApply,
              flow_params: FlowParams,
              log_density: LogDensityByStep,
              step: int) -> Array:
  """Get density difference between current target and push forward of previous.

  Args:
    samples: Array containing samples of shape (batch,) + sample_shape.
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density: function returning the log_density of a sample at given step.
    step: current step.

  Returns:
    deltas: an array containing the difference for each sample.
  """
  transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
  assert_equal_shape([transformed_samples, samples])
  log_density_values_current = log_density(step, transformed_samples)
  log_density_values_previous = log_density(step-1, samples)
  assert_equal_shape([log_density_values_current, log_density_values_previous])
  assert_equal_shape([log_density_values_previous, log_det_jacs])
  deltas = log_density_values_previous - log_density_values_current - log_det_jacs
  return deltas


def get_batch_parallel_free_energy_increment(samples: Array,
                                             flow_apply: FlowApply,
                                             flow_params: FlowParams,
                                             log_density: LogDensityByStep,
                                             step: int) -> Array:
  """Get the log normalizer increments in case where there is no resampling.

  Args:
    samples: (num_batch, num_dim)
    flow_apply: Apply the flow.
    flow_params: Parameters of the flow.
    log_density: Value of the log density.
    step: Step of the algorithm.

  Returns:
    Scalar array containing the increments.
  """
  deltas = get_delta(samples, flow_apply, flow_params, log_density, step)
  chex.assert_rank(deltas, 1)
  # The mean takes the average over the batch. This is equivalent to delaying
  # the average until all temperatures have been accumulated.
  return jnp.mean(deltas)


def transport_free_energy_estimator(samples: Array,
                                    log_weights: Array,
                                    flow_apply: FlowApply,
                                    flow_params: FlowParams,
                                    log_density: LogDensityByStep,
                                    step: int) -> Array:
  """Compute an estimate of the free energy.

  Args:
    samples: Array representing samples (batch,) + sample_shape
    log_weights: scalar representing sample weights (batch,)
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step

  Returns:
    Estimate of the free_energy.
  """
  deltas = get_delta(samples,
                     flow_apply,
                     flow_params,
                     log_density,
                     step)
  assert_equal_shape([deltas, log_weights])
  return jnp.sum(jax.nn.softmax(log_weights) * deltas)


def get_log_normalizer_increment_no_flow(deltas: Array,
                                         log_weights: Array) -> Array:
  assert_equal_shape([deltas, log_weights])
  normalized_log_weights = jax.nn.log_softmax(log_weights)
  total_terms = normalized_log_weights - deltas
  assert_equal_shape([normalized_log_weights, log_weights, total_terms])
  increment = logsumexp(total_terms)
  return increment


def get_log_normalizer_increment(samples: Array,
                                 log_weights: Array,
                                 flow_apply: FlowApply,
                                 flow_params: FlowParams,
                                 log_density: LogDensityByStep,
                                 step: int) -> Array:
  """Get the increment in the log of the normalizing constant estimate.

  Args:
    samples: Array representing samples (batch,) + sample_shape
    log_weights: scalar representing sample weights (batch,)
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step

  Returns:
    Scalar Array, logarithm of normalizing constant increment.
  """
  deltas = get_delta(samples,
                     flow_apply,
                     flow_params,
                     log_density,
                     step)
  increment = get_log_normalizer_increment_no_flow(deltas, log_weights)
  return increment


def reweight_no_flow(log_weights_old: Array,
                     deltas: Array) -> Array:
  log_weights_new_unorm = log_weights_old - deltas
  log_weights_new = jax.nn.log_softmax(log_weights_new_unorm)
  return log_weights_new


def reweight(log_weights_old: Array,
             samples: Array,
             flow_apply: FlowApply,
             flow_params: FlowParams,
             log_density: LogDensityByStep,
             step: int) -> Array:
  """Compute the new weights from the old ones and the deltas.

  Args:
    log_weights_old: scalar representing previous sample weights (batch,)
    samples: Array representing samples (batch,) + sample_shape
    flow_apply: function that applies the flow.
    flow_params: parameters of the flow.
    log_density:  function returning the log_density of a sample at given step.
    step: current step
  Returns:
    logarithm of new weights.
  """
  deltas = get_delta(samples,
                     flow_apply,
                     flow_params,
                     log_density,
                     step)
  log_weights_new = reweight_no_flow(log_weights_old, deltas)
  return log_weights_new


def update_samples_log_weights(
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    flow_params: FlowParams, samples: Array, log_weights: Array, key: RandomKey,
    log_density: LogDensityByStep, step: int, use_resampling: bool,
    use_markov: bool,
    resample_threshold: float) -> Tuple[Array, Array, AcceptanceTuple]:
  """Update samples and log weights once the flow has been learnt."""
  transformed_samples, _ = flow_apply(flow_params, samples)
  assert_equal_shape([transformed_samples, samples])
  log_weights_new = reweight(log_weights, samples, flow_apply, flow_params,
                             log_density, step)
  assert_equal_shape([log_weights_new, log_weights])
  if use_resampling:
    subkey, key = jax.random.split(key)
    resampled_samples, log_weights_resampled = resampling.optionally_resample(
        subkey, log_weights_new, transformed_samples, resample_threshold)
    assert_equal_shape([resampled_samples, transformed_samples])
    assert_equal_shape([log_weights_resampled, log_weights_new])
  else:
    resampled_samples = transformed_samples
    log_weights_resampled = log_weights_new
  if use_markov:
    markov_samples, acceptance_tuple = markov_kernel_apply(
        step, key, resampled_samples)
  else:
    markov_samples = resampled_samples
    acceptance_tuple = (1., 1., 1.)
  return markov_samples, log_weights_resampled, acceptance_tuple
