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

"""Sequential Monte Carlo (SMC) sampler algorithm.

For background see:

Del Moral, Doucet and Jasra. 2006. Sequential Monte Carlo samplers.
Journal of the Royal Statistical Society B.

"""

import time
from typing import Tuple

from absl import logging
from annealed_flow_transport import flow_transport
from annealed_flow_transport import markov_kernel
from annealed_flow_transport import resampling
import annealed_flow_transport.aft_types as tp
import chex
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState


def inner_loop(
    key: RandomKey,
    markov_kernel_apply: MarkovKernelApply,
    samples: Array, log_weights: Array,
    log_density: LogDensityByStep, step: int, config
) -> Tuple[Array, Array, Array, Array]:
  """Inner loop of the algorithm.

  Args:
    key: A JAX random key.
    markov_kernel_apply: functional that applies the Markov transition kernel.
    samples: Array containing samples.
    log_weights: Array containing log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    config: experiment configuration.

  Returns:
    samples_final: samples after the full inner loop has been performed.
    log_weights_final: log_weights after the full inner loop has been performed.
    log_normalizer_increment: Scalar log of normalizing constant increment.
    Acceptance_rates: Acceptance rates of samplers.
  """

  deltas = flow_transport.get_delta_no_flow(samples, log_density, step)
  log_normalizer_increment = flow_transport.get_log_normalizer_increment_no_flow(
      deltas, log_weights)
  log_weights_new = flow_transport.reweight_no_flow(log_weights, deltas)
  if config.use_resampling:
    subkey, key = jax.random.split(key)
    resampled_samples, log_weights_resampled = resampling.optionally_resample(
        subkey, log_weights_new, samples, config.resample_threshold)
    assert_equal_shape([resampled_samples, samples])
    assert_equal_shape([log_weights_resampled, log_weights_new])
  else:
    resampled_samples = samples
    log_weights_resampled = log_weights_new
  markov_samples, acceptance_tuple = markov_kernel_apply(
      step, key, resampled_samples)

  return markov_samples, log_weights_resampled, log_normalizer_increment, acceptance_tuple


def get_short_inner_loop(markov_kernel_by_step: MarkovKernelApply,
                         density_by_step: LogDensityByStep,
                         config):
  """Get a short version of inner loop."""
  def short_inner_loop(rng_key: RandomKey,
                       loc_samples: Array,
                       loc_log_weights: Array,
                       loc_step: int):
    return inner_loop(rng_key,
                      markov_kernel_by_step,
                      loc_samples,
                      loc_log_weights,
                      density_by_step,
                      loc_step,
                      config)
  return short_inner_loop


def fast_outer_loop_smc(density_by_step: LogDensityByStep,
                        initial_sampler: InitialSampler,
                        markov_kernel_by_step: MarkovKernelApply,
                        key: RandomKey,
                        config) -> ParticleState:
  """A fast SMC loop for evaluation or use inside other algorithms."""
  key, subkey = jax.random.split(key)

  samples = initial_sampler(subkey, config.batch_size, config.sample_shape)
  log_weights = -jnp.log(config.batch_size) * jnp.ones(config.batch_size)
  short_inner_loop = get_short_inner_loop(markov_kernel_by_step,
                                          density_by_step, config)

  keys = jax.random.split(key, config.num_temps-1)

  def scan_step(passed_state, per_step_input):
    samples, log_weights = passed_state
    current_step, current_key = per_step_input
    new_samples, new_log_weights, log_z_increment, _ = short_inner_loop(
        current_key, samples, log_weights, current_step)
    new_passed_state = (new_samples, new_log_weights)
    return new_passed_state, log_z_increment

  init_state = (samples, log_weights)
  per_step_inputs = (np.arange(1, config.num_temps), keys)
  final_state, log_normalizer_increments = jax.lax.scan(scan_step,
                                                        init_state,
                                                        per_step_inputs
                                                        )
  log_normalizer_estimate = jnp.sum(log_normalizer_increments)
  particle_state = ParticleState(
      samples=final_state[0],
      log_weights=final_state[1],
      log_normalizer_estimate=log_normalizer_estimate)
  return particle_state


def outer_loop_smc(initial_log_density: LogDensityNoStep,
                   final_log_density: LogDensityNoStep,
                   initial_sampler: InitialSampler,
                   key: RandomKey,
                   config) -> AlgoResultsTuple:
  """The outer loop for Annealed Flow Transport Monte Carlo.

  Args:
    initial_log_density: The log density of the starting distribution.
    final_log_density: The log density of the target distribution.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    config: A ConfigDict containing the configuration.

  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
  num_temps = config.num_temps
  density_by_step = flow_transport.GeometricAnnealingSchedule(
      initial_log_density, final_log_density, num_temps)
  markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
      config.mcmc_config, density_by_step, num_temps)

  key, subkey = jax.random.split(key)

  samples = initial_sampler(subkey, config.batch_size, config.sample_shape)
  log_weights = -jnp.log(config.batch_size) * jnp.ones(config.batch_size)

  logging.info('Jitting step...')
  inner_loop_jit = jax.jit(
      get_short_inner_loop(markov_kernel_by_step, density_by_step, config))

  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time.time()
  inner_loop_jit(key, samples, log_weights, 1)
  initial_finish_time = time.time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info('Initial step time / seconds  %f: ', initial_time_diff)
  logging.info('Launching training...')
  log_normalizer_estimate = 0.
  start_time = time.time()
  for step in range(1, num_temps):
    subkey, key = jax.random.split(key)
    samples, log_weights, log_normalizer_increment, acceptance = inner_loop_jit(
        subkey, samples, log_weights, step)
    acceptance_nuts = float(np.asarray(acceptance[0]))
    acceptance_hmc = float(np.asarray(acceptance[1]))
    acceptance_rwm = float(np.asarray(acceptance[2]))
    log_normalizer_estimate += log_normalizer_increment
    if step % config.report_step == 0:
      beta = density_by_step.get_beta(step)
      logging.info(
          'Step %05d: beta %f Acceptance rate NUTS %f Acceptance rate HMC %f Acceptance rate RWM %f',
          step, beta, acceptance_nuts, acceptance_hmc, acceptance_rwm)

  finish_time = time.time()
  delta_time = finish_time - start_time
  logging.info('Delta time / seconds  %f: ', delta_time)
  logging.info('Log normalizer estimate %f: ', log_normalizer_estimate)
  results = AlgoResultsTuple(
      test_samples=samples,
      test_log_weights=log_weights,
      log_normalizer_estimate=log_normalizer_estimate,
      delta_time=delta_time,
      initial_time_diff=initial_time_diff)
  return results
