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

"""Annealed Flow Transport (AFT) Monte Carlo algorithm.

For more detail see:

Arbel, Matthews and Doucet. 2021. Annealed Flow Transport Monte Carlo.
International Conference on Machine Learning.

"""

import time
from typing import NamedTuple, Tuple

from absl import logging
from annealed_flow_transport import flow_transport
from annealed_flow_transport import markov_kernel
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import numpy as np
import optax

Array = jnp.ndarray
UpdateFn = tp.UpdateFn
OptState = tp.OptState
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
SamplesTuple = tp.SamplesTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
MarkovKernelApply = tp.MarkovKernelApply
FreeEnergyEval = tp.FreeEnergyEval
VfesTuple = tp.VfesTuple
LogDensityByStep = tp.LogDensityByStep
AcceptanceTuple = tp.AcceptanceTuple
LogWeightsTuple = tp.LogWeightsTuple
AlgoResultsTuple = tp.AlgoResultsTuple


def get_initial_samples_log_weight_tuples(
    initial_sampler: InitialSampler, key: RandomKey,
    config) -> Tuple[SamplesTuple, LogWeightsTuple]:
  """Get initial train/validation/test state depending on config."""
  batch_sizes = (config.estimation_batch_size,
                 config.estimation_batch_size,
                 config.batch_size)
  subkeys = jax.random.split(key, 3)
  samples_tuple = SamplesTuple(*[
      initial_sampler(elem, batch, config.sample_shape)
      for elem, batch in zip(subkeys, batch_sizes)
  ])
  log_weights_tuple = LogWeightsTuple(*[-jnp.log(batch) * jnp.ones(
      batch) for batch in  batch_sizes])
  return samples_tuple, log_weights_tuple


def update_tuples(
    samples_tuple: SamplesTuple, log_weights_tuple: LogWeightsTuple,
    key: RandomKey, flow_apply: FlowApply, flow_params: FlowParams,
    markov_kernel_apply: MarkovKernelApply, log_density: LogDensityByStep,
    step: int, config) -> Tuple[SamplesTuple, LogWeightsTuple, AcceptanceTuple]:
  """Update the samples and log weights and return diagnostics."""
  samples_list = []
  log_weights_list = []
  acceptance_tuple_list = []
  subkeys = jax.random.split(key, 3)
  for curr_samples, curr_log_weights, subkey in zip(samples_tuple,
                                                    log_weights_tuple,
                                                    subkeys):
    new_samples, new_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
        flow_apply=flow_apply,
        markov_kernel_apply=markov_kernel_apply,
        flow_params=flow_params,
        samples=curr_samples,
        log_weights=curr_log_weights,
        key=subkey,
        log_density=log_density,
        step=step,
        use_resampling=config.use_resampling,
        use_markov=config.use_markov,
        resample_threshold=config.resample_threshold)
    samples_list.append(new_samples)
    log_weights_list.append(new_log_weights)
    acceptance_tuple_list.append(acceptance_tuple)
  samples_tuple = SamplesTuple(*samples_list)
  log_weights_tuple = LogWeightsTuple(*log_weights_list)
  test_acceptance_tuple = acceptance_tuple_list[-1]
  return samples_tuple, log_weights_tuple, test_acceptance_tuple


class OptimizationLoopState(NamedTuple):
  opt_state: OptState
  flow_params: FlowParams
  inner_step: int
  opt_vfes: VfesTuple
  best_params: FlowParams
  best_validation_vfe: Array
  best_index: int


def flow_estimate_step(loop_state: OptimizationLoopState,
                       free_energy_and_grad: FreeEnergyAndGrad,
                       train_samples: Array, train_log_weights: Array,
                       outer_step: int, validation_samples: Array,
                       validation_log_weights: Array,
                       free_energy_eval: FreeEnergyEval,
                       opt_update: UpdateFn) -> OptimizationLoopState:
  """A single step of the flow estimation loop."""
  # Evaluate the flow on train and validation particles.
  train_vfe, flow_grads = free_energy_and_grad(loop_state.flow_params,
                                               train_samples,
                                               train_log_weights,
                                               outer_step)
  validation_vfe = free_energy_eval(loop_state.flow_params,
                                    validation_samples,
                                    validation_log_weights,
                                    outer_step)

  # Update the best parameters, best validation vfe and index
  # if the measured validation vfe is better.
  validation_vfe_is_better = validation_vfe < loop_state.best_validation_vfe
  new_best_params = jax.lax.cond(validation_vfe_is_better,
                                 lambda _: loop_state.flow_params,
                                 lambda _: loop_state.best_params,
                                 operand=None)
  new_best_validation_vfe = jnp.where(validation_vfe_is_better,
                                      validation_vfe,
                                      loop_state.best_validation_vfe)
  new_best_index = jnp.where(validation_vfe_is_better,
                             loop_state.inner_step,
                             loop_state.best_index)

  # Update the logs of train and validation vfes.
  new_train_vfes = loop_state.opt_vfes.train_vfes.at[loop_state.inner_step].set(
      train_vfe)
  new_validation_vfes = loop_state.opt_vfes.validation_vfes.at[
      loop_state.inner_step].set(validation_vfe)

  new_opt_vfes = VfesTuple(train_vfes=new_train_vfes,
                           validation_vfes=new_validation_vfes)

  # Apply gradients ready for next round of flow evaluations in the next step.
  updates, new_opt_state = opt_update(flow_grads,
                                      loop_state.opt_state)
  new_flow_params = optax.apply_updates(loop_state.flow_params,
                                        updates)
  new_inner_step = loop_state.inner_step + 1

  # Pack everything into the next loop state.
  new_state_tuple = OptimizationLoopState(new_opt_state, new_flow_params,
                                          new_inner_step, new_opt_vfes,
                                          new_best_params,
                                          new_best_validation_vfe,
                                          new_best_index)
  return new_state_tuple


def flow_estimation_should_continue(loop_state: OptimizationLoopState,
                                    opt_iters: int,
                                    stopping_criterion: str) -> bool:
  """Based on stopping criterion control termination of flow estimation."""
  if stopping_criterion == 'time':
    return loop_state.inner_step < opt_iters
  elif stopping_criterion == 'greedy_time':
    index = loop_state.inner_step
    best_index = loop_state.best_index
    return jnp.logical_and(best_index == index-1, index < opt_iters)
  else:
    raise NotImplementedError


def optimize_free_energy(
    opt_update: UpdateFn, opt_init_state: OptState,
    flow_init_params: FlowParams, free_energy_and_grad: FreeEnergyAndGrad,
    free_energy_eval: FreeEnergyEval, train_samples: Array,
    train_log_weights: Array, validation_samples: Array,
    validation_log_weights: Array, outer_step: int, opt_iters: int,
    stopping_criterion: str) -> Tuple[FlowParams, VfesTuple]:
  """Optimize an estimate of the free energy.

  Args:
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial parameters of the flow.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    free_energy_eval: function giving estimate of free energy only.
    train_samples: Array of shape (batch,)+sample_shape
    train_log_weights: Array of shape (batch,)
    validation_samples: Array of shape (batch,)
    validation_log_weights: Array of shape (batch,)
    outer_step: int giving current outer step of algorithm.
    opt_iters: number of flow estimation iters.
    stopping_criterion: One of 'time' or 'greedy-time'.

  Returns:
    flow_params: optimized flow parameters.
    free_energies: array containing all estimates of free energy.
  """
  opt_state = opt_init_state
  flow_params = flow_init_params
  train_vfes = jnp.zeros(opt_iters)
  validation_vfes = jnp.zeros(opt_iters)
  opt_vfes = VfesTuple(train_vfes, validation_vfes)

  def body_fun(loop_state: OptimizationLoopState) -> OptimizationLoopState:
    return flow_estimate_step(loop_state, free_energy_and_grad, train_samples,
                              train_log_weights, outer_step, validation_samples,
                              validation_log_weights, free_energy_eval,
                              opt_update)

  def cond_fun(loop_state: OptimizationLoopState) -> bool:
    return flow_estimation_should_continue(loop_state, opt_iters,
                                           stopping_criterion)

  initial_loop_state = OptimizationLoopState(opt_state, flow_params, 0,
                                             opt_vfes, flow_params, jnp.inf, -1)
  final_loop_state = jax.lax.while_loop(cond_fun,
                                        body_fun,
                                        initial_loop_state)
  return final_loop_state.best_params, final_loop_state.opt_vfes


def inner_loop(
    key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
    free_energy_eval: FreeEnergyEval, opt_update: UpdateFn,
    opt_init_state: OptState, flow_init_params: FlowParams,
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    samples_tuple: SamplesTuple, log_weights_tuple: LogWeightsTuple,
    log_density: LogDensityByStep, step: int, config
) -> Tuple[FlowParams, OptState, VfesTuple, Array, AcceptanceTuple]:
  """Inner loop of the algorithm.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    free_energy_eval: function giving estimate of free energy only.
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial parameters of the flow.
    flow_apply: function that applies the flow.
    markov_kernel_apply: functional that applies the Markov transition kernel.
    samples_tuple: Tuple containing train/validation/test samples.
    log_weights_tuple: Tuple containing train/validation/test log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    config: experiment configuration.

  Returns:
    samples_final: samples after the full inner loop has been performed.
    log_weights_final: log_weights after the full inner loop has been performed.
    free_energies: array containing all estimates of free energy.
    log_normalizer_increment: Scalar log of normalizing constant increment.
  """
  flow_params, vfes_tuple = optimize_free_energy(
      opt_update=opt_update,
      opt_init_state=opt_init_state,
      flow_init_params=flow_init_params,
      free_energy_and_grad=free_energy_and_grad,
      free_energy_eval=free_energy_eval,
      train_samples=samples_tuple.train_samples,
      train_log_weights=log_weights_tuple.train_log_weights,
      validation_samples=samples_tuple.validation_samples,
      validation_log_weights=log_weights_tuple.validation_log_weights,
      outer_step=step,
      opt_iters=config.optimization_config.free_energy_iters,
      stopping_criterion=config.stopping_criterion)
  log_normalizer_increment = flow_transport.get_log_normalizer_increment(
      samples_tuple.test_samples, log_weights_tuple.test_log_weights,
      flow_apply, flow_params, log_density, step)

  samples_tuple, log_weights_tuple, test_acceptance_tuple = update_tuples(
      samples_tuple=samples_tuple,
      log_weights_tuple=log_weights_tuple,
      key=key,
      flow_apply=flow_apply,
      flow_params=flow_params,
      markov_kernel_apply=markov_kernel_apply,
      log_density=log_density,
      step=step,
      config=config)

  return samples_tuple, log_weights_tuple, vfes_tuple, log_normalizer_increment, test_acceptance_tuple


def outer_loop_aft(opt_update: UpdateFn,
                   opt_init_state: OptState,
                   flow_init_params: FlowParams,
                   flow_apply: FlowApply,
                   initial_log_density: LogDensityNoStep,
                   final_log_density: LogDensityNoStep,
                   initial_sampler: InitialSampler,
                   key: RandomKey,
                   config,
                   log_step_output) -> AlgoResultsTuple:
  """The outer loop for Annealed Flow Transport Monte Carlo.

  Args:
    opt_update: A Optax optimizer update function.
    opt_init_state: Optax initial state.
    flow_init_params: Initial parameters for the flow.
    flow_apply: Function that evaluates flow on parameters and samples.
    initial_log_density: The log density of the starting distribution.
    final_log_density: The log density of the target distribution.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    config: A ConfigDict containing the configuration.
    log_step_output: Function to log step output or None.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
  num_temps = config.num_temps
  density_by_step = flow_transport.GeometricAnnealingSchedule(
      initial_log_density, final_log_density, num_temps)
  markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
      config.mcmc_config, density_by_step, num_temps)

  def free_energy_short(flow_params: FlowParams,
                        samples: Array,
                        log_weights: Array,
                        step: int) -> Array:
    return flow_transport.transport_free_energy_estimator(
        samples, log_weights, flow_apply, flow_params, density_by_step, step)

  free_energy_eval = jax.jit(free_energy_short)
  free_energy_and_grad = jax.value_and_grad(free_energy_short)
  key, subkey = jax.random.split(key)

  samples_tuple, log_weights_tuple = get_initial_samples_log_weight_tuples(
      initial_sampler, subkey, config)

  def short_inner_loop(rng_key: RandomKey,
                       loc_samples_tuple: SamplesTuple,
                       loc_log_weights_tuple: LogWeightsTuple,
                       loc_step: int):
    return inner_loop(key=rng_key,
                      free_energy_and_grad=free_energy_and_grad,
                      free_energy_eval=free_energy_eval,
                      opt_update=opt_update,
                      opt_init_state=opt_init_state,
                      flow_init_params=flow_init_params,
                      flow_apply=flow_apply,
                      markov_kernel_apply=markov_kernel_by_step,
                      samples_tuple=loc_samples_tuple,
                      log_weights_tuple=loc_log_weights_tuple,
                      log_density=density_by_step,
                      step=loc_step,
                      config=config)
  logging.info('Jitting step...')
  inner_loop_jit = jax.jit(short_inner_loop)

  opt_iters = config.optimization_config.free_energy_iters
  if log_step_output is not None:
    zero_vfe_tuple = VfesTuple(train_vfes=jnp.zeros(opt_iters),
                               validation_vfes=jnp.zeros(opt_iters))
    log_step_output(samples_tuple, log_weights_tuple, zero_vfe_tuple, 0., 1.,
                    1.)
  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time.time()
  inner_loop_jit(key, samples_tuple, log_weights_tuple, 1)
  initial_finish_time = time.time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info('Initial step time / seconds  %f: ', initial_time_diff)
  logging.info('Launching training...')
  log_normalizer_estimate = 0.
  start_time = time.time()
  for step in range(1, num_temps):
    subkey, key = jax.random.split(key)
    samples_tuple, log_weights_tuple, vfes_tuple, log_normalizer_increment, test_acceptance = inner_loop_jit(
        subkey, samples_tuple, log_weights_tuple, step)
    acceptance_nuts = float(np.asarray(test_acceptance[0]))
    acceptance_hmc = float(np.asarray(test_acceptance[1]))
    acceptance_rwm = float(np.asarray(test_acceptance[2]))
    log_normalizer_estimate += log_normalizer_increment
    if step % config.report_step == 0:
      beta = density_by_step.get_beta(step)
      logging.info(
          'Step %05d: beta %f Acceptance rate NUTS %f Acceptance rate HMC %f Acceptance rate RWM %f',
          step, beta, acceptance_nuts, acceptance_hmc, acceptance_rwm
          )
      if log_step_output is not None:
        log_step_output(samples_tuple, log_weights_tuple,
                        vfes_tuple, log_normalizer_increment, acceptance_nuts,
                        acceptance_hmc)
  finish_time = time.time()
  delta_time = finish_time - start_time
  logging.info('Delta time / seconds  %f: ', delta_time)
  logging.info('Log normalizer estimate %f: ', log_normalizer_estimate)
  results = AlgoResultsTuple(
      test_samples=samples_tuple.test_samples,
      test_log_weights=log_weights_tuple.test_log_weights,
      log_normalizer_estimate=log_normalizer_estimate,
      delta_time=delta_time,
      initial_time_diff=initial_time_diff)
  return results
