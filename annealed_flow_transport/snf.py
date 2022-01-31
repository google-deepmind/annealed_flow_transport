# Copyright 2021 DeepMind Technologies Limited.
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

"""Stochastic Normalizing Flows as implemented in Wu et al. 2020.

For fixed flows this is equivalent to Annealed Importance Sampling with flows,
and without resampling.

Training is then based on the corresponding ELBO.

This is not reparameterizable using a continuous function but Wu et al.
proceed as if it where using a "straight through" gradient estimator.
"""
import time

from absl import logging
from annealed_flow_transport import flow_transport
from annealed_flow_transport import markov_kernel
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import optax


Array = jnp.ndarray
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
SamplesTuple = tp.SamplesTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
AcceptanceTuple = tp.AcceptanceTuple
LogWeightsTuple = tp.LogWeightsTuple
AlgoResultsTuple = tp.AlgoResultsTuple


def inner_loop_snf(key: RandomKey,
                   transition_params: FlowParams, flow_apply: FlowApply,
                   markov_kernel_apply: MarkovKernelApply,
                   initial_sampler: InitialSampler,
                   log_density: LogDensityByStep, config):
  """Inner loop of Stochastic Normalizing Flows.

  Uses Scan step requiring trees that have the same structure as the base input
  but with each leaf extended with an extra array index of size num_transitions.
  We call this an extended tree.

  Args:
    key: A JAX random key.
    transition_params: Extended tree of flow parameters.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    initial_sampler: A function that produces the initial samples.
    log_density: A function evaluating the log density for each step.
    config: A ConfigDict containing the configuration.
  Returns:
    vfe: variational free energy.
  """
  subkey, key = jax.random.split(key)
  initial_samples = initial_sampler(subkey, config.snf_batch_size,
                                    config.sample_shape)
  initial_log_weights = -jnp.log(config.snf_batch_size) * jnp.ones(
      config.snf_batch_size)

  def scan_step(passed_state, per_step_input):
    samples, log_weights = passed_state
    flow_params, key, inner_step = per_step_input
    vfe_increment = flow_transport.get_batch_parallel_free_energy_increment(
        samples=samples,
        flow_apply=flow_apply,
        flow_params=flow_params,
        log_density=log_density,
        step=inner_step)
    next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
        flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
        flow_params=flow_params, samples=samples, log_weights=log_weights,
        key=key, log_density=log_density, step=inner_step, use_resampling=False,
        use_markov=config.use_markov,
        resample_threshold=config.resample_threshold)
    next_passed_state = (next_samples, next_log_weights)
    per_step_output = (vfe_increment, acceptance_tuple)
    return next_passed_state, per_step_output

  initial_state = (initial_samples, initial_log_weights)
  inner_steps = jnp.arange(1, config.num_temps)
  keys = jax.random.split(key, config.num_temps-1)
  per_step_inputs = (transition_params, keys, inner_steps)
  unused_final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state,
                                                      per_step_inputs)
  vfe_increments, unused_acceptance_tuples = per_step_outputs
  vfe = jnp.sum(vfe_increments)
  return vfe


def outer_loop_snf(flow_init_params: FlowParams,
                   flow_apply: FlowApply,
                   initial_log_density: LogDensityNoStep,
                   final_log_density: LogDensityNoStep,
                   initial_sampler: InitialSampler,
                   key: RandomKey,
                   opt,
                   config,
                   log_step_output,
                   save_checkpoint):
  """Outer loop for Stochastic Normalizing Flows.

  Args:
    flow_init_params: initial state of the flow.
    flow_apply: function that applies the flow.
    initial_log_density: The log density of the starting distribution.
    final_log_density: The log density of the target distribution.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    opt: An Optax optimizer.
    config: A ConfigDict containing the configuration.
    log_step_output: Callable that logs the step output.
    save_checkpoint: None or function that takes params and saves them.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
  num_temps = config.num_temps
  density_by_step = flow_transport.GeometricAnnealingSchedule(
      initial_log_density, final_log_density, num_temps)
  markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
      config.mcmc_config, density_by_step, num_temps)

  def short_inner_loop(curr_transition_params,
                       rng_key: RandomKey):
    return inner_loop_snf(rng_key,
                          curr_transition_params, flow_apply,
                          markov_kernel_by_step, initial_sampler,
                          density_by_step, config)

  repeater = lambda x: jnp.repeat(x[None], num_temps-1, axis=0)
  transition_params = jax.tree_util.tree_map(repeater, flow_init_params)

  opt_state = opt.init(transition_params)

  def vi_update(curr_key, curr_transition_params, curr_opt_state):
    subkey, curr_key = jax.random.split(curr_key)
    objective, flow_grads = jax.value_and_grad(short_inner_loop)(
        curr_transition_params, subkey)
    updates, new_opt_state = opt.update(flow_grads,
                                        curr_opt_state)
    new_transition_params = optax.apply_updates(curr_transition_params,
                                                updates)
    return curr_key, new_transition_params, objective, new_opt_state
  jit_vi_update = jax.jit(vi_update)

  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time.time()
  jit_vi_update(key, transition_params, opt_state)
  initial_finish_time = time.time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info('Initial step time / seconds  %f: ', initial_time_diff)
  logging.info('Launching training...')
  place_holder_array = jnp.array([0.])

  start_time = time.time()
  for step in range(config.snf_num_iters):
    key, transition_params, vfe, opt_state = jit_vi_update(
        key, transition_params, opt_state)
    if step % config.report_step == 0:
      if log_step_output is not None:
        delta_time = time.time()-start_time
        log_step_output(step=step,
                        training_objective=vfe,
                        log_normalizer_estimate=-1.*vfe,
                        delta_time=delta_time,
                        samples=place_holder_array,
                        log_weights=place_holder_array)
      logging.info(
          'Step %05d: Free energy %f',
          step, vfe
          )

  finish_time = time.time()
  delta_time = finish_time - start_time
  final_log_normalizer_estimate = -1.*vfe
  logging.info('Delta time / seconds  %f: ', delta_time)
  logging.info('Log normalizer estimate %f: ', final_log_normalizer_estimate)

  if save_checkpoint:
    save_checkpoint(transition_params)
  results = AlgoResultsTuple(
      test_samples=place_holder_array,
      test_log_weights=place_holder_array,
      log_normalizer_estimate=final_log_normalizer_estimate,
      delta_time=delta_time,
      initial_time_diff=initial_time_diff)
  return results
