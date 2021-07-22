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

"""Code for Markov transition kernels."""

from typing import Tuple

import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

mcmc = tfp.mcmc
ConfigDict = tp.ConfigDict
Array = tp.Array
LogDensityByStep = tp.LogDensityByStep
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply


class InterpolatedStepSize(object):
  """Interpolate MCMC step sizes."""

  def __init__(self,
               config: ConfigDict,
               total_num_time_steps: int):
    self._config = config
    self._total_num_time_steps = total_num_time_steps

  def __call__(self, time_step: int):
    final_step = self._total_num_time_steps-1.
    beta = time_step/final_step
    return jnp.interp(beta,
                      jnp.array(self._config.step_times),
                      jnp.array(self._config.step_sizes))


class MarkovTransitionKernel(object):
  """Wraps TFP slice sampling and NUTS allowing configuration/composition."""

  def __init__(self,
               config: ConfigDict,
               density_by_step: LogDensityByStep,
               total_time_steps: int):
    self._config = config
    self._density_by_step = density_by_step
    if hasattr(config, 'slice_step_config'):
      self._slice_step_size = InterpolatedStepSize(
          config.slice_step_config,
          total_time_steps)
    if hasattr(config, 'hmc_step_config'):
      self._hmc_step_size = InterpolatedStepSize(
          config.hmc_step_config,
          total_time_steps)
    if hasattr(config, 'nuts_step_config'):
      self._nuts_step_size = InterpolatedStepSize(
          config.nuts_step_config,
          total_time_steps)

  def _add_markov_kernel(self, samples_in: Array,
                         key: RandomKey,
                         outer_step: int,
                         num_transitions: int,
                         tfp_kernel,
                         trace_fn) -> Tuple[Array, Array]:
    """Add a given tfp Markov kernel to the sequence."""
    results = mcmc.sample_chain(
        num_results=1,
        num_burnin_steps=num_transitions - 1,
        current_state=samples_in[0, :, :],
        kernel=tfp_kernel(outer_step),
        trace_fn=trace_fn,
        seed=key)
    if trace_fn is None:  # Corresponds to slice sampler which always accepts.
      return results, 1.
    else:
      samples = results.all_states
      acceptance = jnp.mean(results.trace)
      return samples, acceptance

  def __call__(self, step: int, key: RandomKey, samples_in: Array) -> Array:
    """A single step of slice sampling followed by NUTS.

    Args:
      step: The time step of the overall algorithm.
      key: A JAX random key.
      samples_in: The current samples.
    Returns:
      New samples.
    """
    samples = samples_in[None, :, :]  # Add axis to match tfp format.
    log_density_lambda = lambda x: self._density_by_step(step, x)
    shared_trace_fn = lambda _, pkr: pkr.is_accepted
    def slice_sampling_tfp_kernel(curr_step):
      kernel = mcmc.SliceSampler(
          target_log_prob_fn=log_density_lambda,
          step_size=self._slice_step_size(curr_step),
          max_doublings=self._config.slice_max_doublings)
      return kernel

    def nuts_tfp_kernel(curr_step):
      kernel = mcmc.NoUTurnSampler(
          target_log_prob_fn=log_density_lambda,
          step_size=self._nuts_step_size(curr_step),
          max_tree_depth=self._config.nuts_max_tree_depth)
      return kernel

    def hmc_tfp_kernel(curr_step):
      kernel = mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=log_density_lambda,
          step_size=self._hmc_step_size(curr_step),
          num_leapfrog_steps=self._config.hmc_num_leapfrog_steps)
      return kernel

    if self._config.slice_steps_per_iter != 0:
      subkey, key = jax.random.split(key)
      samples, unused_acc = self._add_markov_kernel(
          samples, subkey, step, self._config.slice_steps_per_iter,
          slice_sampling_tfp_kernel, None)
    if self._config.nuts_steps_per_iter != 0:
      subkey, key = jax.random.split(key)
      samples, nuts_acc = self._add_markov_kernel(
          samples, subkey, step, self._config.nuts_steps_per_iter,
          nuts_tfp_kernel, shared_trace_fn)
    else:
      nuts_acc = 1.

    if self._config.hmc_steps_per_iter != 0:
      samples, hmc_acc = self._add_markov_kernel(
          samples, key, step, self._config.hmc_steps_per_iter,
          hmc_tfp_kernel, shared_trace_fn)
    else:
      hmc_acc = 1.

    acceptance_tuple = (nuts_acc, hmc_acc)
    samples_out = samples[0, :, :]  # Remove axis to match format in this code.
    return samples_out, acceptance_tuple
