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

"""Code for variational inference (VI) with normalizing flows.

For background see:

Rezende and Mohamed. 2015. Variational Inference with Normalizing Flows.
International Conference of Machine Learning.

"""

from absl import logging
import annealed_flow_transport.aft_types as tp
import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
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
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState


def vfe_naive_importance(initial_sampler: InitialSampler,
                         initial_density: LogDensityNoStep,
                         final_density: LogDensityNoStep,
                         flow_apply: FlowApply,
                         flow_params: FlowParams,
                         key: RandomKey,
                         config) -> ParticleState:
  """Estimate log normalizing constant using naive importance sampling."""
  samples = initial_sampler(key,
                            config.batch_size,
                            config.sample_shape)
  transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
  assert_equal_shape([transformed_samples, samples])
  log_density_target = final_density(transformed_samples)
  log_density_initial = initial_density(samples)
  assert_equal_shape([log_density_initial, log_density_target])
  log_density_approx = log_density_initial - log_det_jacs
  log_importance_weights = log_density_target - log_density_approx
  log_normalizer_estimate = logsumexp(log_importance_weights) - np.log(
      config.batch_size)
  particle_state = ParticleState(
      samples=transformed_samples,
      log_weights=log_importance_weights,
      log_normalizer_estimate=log_normalizer_estimate)
  return particle_state


def vi_free_energy(flow_params: FlowParams,
                   key: RandomKey,
                   initial_sampler: InitialSampler,
                   initial_density: LogDensityNoStep,
                   final_density: LogDensityNoStep,
                   flow_apply: FlowApply,
                   config):
  """The variational free energy used in VI with normalizing flows."""
  samples = initial_sampler(key,
                            config.batch_size,
                            config.sample_shape)
  transformed_samples, log_det_jacs = flow_apply(flow_params, samples)
  assert_equal_shape([transformed_samples, samples])
  log_density_target = final_density(transformed_samples)
  log_density_initial = initial_density(samples)
  assert_equal_shape([log_density_initial, log_density_target])
  log_density_approx = log_density_initial - log_det_jacs
  assert_equal_shape([log_density_approx, log_density_initial])
  free_energies = log_density_approx - log_density_target
  free_energy = jnp.mean(free_energies)
  return free_energy


def outer_loop_vi(initial_sampler: InitialSampler,
                  opt_update: UpdateFn,
                  opt_init_state: OptState,
                  flow_init_params: FlowParams,
                  flow_apply: FlowApply,
                  key: RandomKey,
                  initial_log_density: LogDensityNoStep,
                  final_log_density: LogDensityNoStep,
                  config,
                  save_checkpoint) -> AlgoResultsTuple:
  """The training loop for variational inference with normalizing flows.

  Args:
    initial_sampler: Produces samples from the base distribution.
    opt_update: Optax update function for the optimizer.
    opt_init_state: Optax initial state for the optimizer.
    flow_init_params: Initial params for the flow.
    flow_apply: A callable that evaluates the flow for given params and samples.
    key: A Jax random Key.
    initial_log_density: Function that evaluates the base density.
    final_log_density: Function that evaluates the target density.
    config: configuration ConfigDict.
    save_checkpoint: None or function that takes params and saves them.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """

  def vi_free_energy_short(loc_flow_params,
                           loc_key):
    return vi_free_energy(loc_flow_params,
                          loc_key,
                          initial_sampler,
                          initial_log_density,
                          final_log_density,
                          flow_apply,
                          config)

  free_energy_and_grad = jax.jit(jax.value_and_grad(vi_free_energy_short))
  flow_params = flow_init_params
  opt_state = opt_init_state

  def vi_update(curr_key, curr_flow_params, curr_opt_state):
    subkey, curr_key = jax.random.split(curr_key)
    new_free_energy, flow_grads = free_energy_and_grad(curr_flow_params,
                                                       subkey)
    updates, new_opt_state = opt_update(flow_grads,
                                        curr_opt_state)
    new_flow_params = optax.apply_updates(curr_flow_params,
                                          updates)
    return curr_key, new_flow_params, new_free_energy, new_opt_state

  jit_vi_update = jax.jit(vi_update)

  step = 0
  while step < config.vi_iters:
    key, flow_params, curr_free_energy, opt_state = jit_vi_update(key,
                                                                  flow_params,
                                                                  opt_state)
    if step % config.vi_report_step == 0:
      logging.info('Step %05d: free_energy %f:', step, curr_free_energy)
      if config.vi_estimator == 'elbo':
        log_normalizer_estimate = -1.*curr_free_energy
      elif config.vi_estimator == 'importance':
        subkey, key = jax.random.split(key, 2)
        particle_state = vfe_naive_importance(
            initial_sampler, initial_log_density, final_log_density, flow_apply,
            flow_params, subkey, config)
        log_normalizer_estimate = particle_state.log_normalizer_estimate
      else:
        raise NotImplementedError
      logging.info('Log normalizer estimate %f:', log_normalizer_estimate)

    step += 1

  if save_checkpoint:
    save_checkpoint(flow_params)

  particle_state = vfe_naive_importance(initial_sampler, initial_log_density,
                                        final_log_density, flow_apply,
                                        flow_params, key, config)
  results = AlgoResultsTuple(
      test_samples=particle_state.samples,
      test_log_weights=particle_state.log_weights,
      log_normalizer_estimate=particle_state.log_normalizer_estimate,
      delta_time=0.,  # These are currently set with placeholders.
      initial_time_diff=0.)
  return results
