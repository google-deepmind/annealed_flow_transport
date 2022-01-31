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
import chex
import jax
import jax.numpy as jnp
import numpy as np
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


def random_walk_metropolis_tfp_compat(tfp_samples_in: Array,
                                      proposal_scale: Array,
                                      log_density_by_step: LogDensityByStep,
                                      temp_step: int, num_mh_steps: int,
                                      key: RandomKey) -> Tuple[Array, Array]:
  """Random walk Metropolis-Hastings algorithm. Shapes wrapped for TFP compat.

  Args:
    tfp_samples_in: (1, num_batch, num_dim)
    proposal_scale: Scalar representing scale of isotropic normal proposal.
    log_density_by_step: Target log density.
    temp_step: Step of outer annealing algorithm.
    num_mh_steps: Number of Metropolis-Hastings steps.
    key: Jax Random Key.
  Returns:
    samples_out_tfp: (1, num_batch, num_dim)
    acceptance: Average acceptance rate of chains.
  """
  chex.assert_shape(tfp_samples_in, (1, None, None))
  samples_in = tfp_samples_in[0, :, :]
  samples_out, acceptance_rate = random_walk_metropolis(samples_in,
                                                        proposal_scale,
                                                        log_density_by_step,
                                                        temp_step, num_mh_steps,
                                                        key)
  samples_out_tfp = samples_out[None, :, :]
  return samples_out_tfp, acceptance_rate


def random_walk_metropolis(samples_in: Array, proposal_scale: Array,
                           log_density_by_step: LogDensityByStep,
                           temp_step: int, num_mh_steps: int,
                           key: RandomKey) -> Tuple[Array, Array]:
  """Corrected random walk Metropolis-Hastings algorithm.

  Args:
    samples_in: (num_batch, num_dim)
    proposal_scale: Scalar representing scale of isotropic normal proposal.
    log_density_by_step: Target log density.
    temp_step: Step of outer annealing algorithm.
    num_mh_steps: Number of Metropolis-Hastings steps.
    key: Jax Random Key.
  Returns:
    samples_out: (num_batch, num_dim)
    acceptance: Average acceptance rate of chains.
  """
  chex.assert_rank((samples_in, proposal_scale), (2, 0))
  num_batch, num_dim = np.shape(samples_in)
  def rwm_step(previous_samples: Array, curr_key: RandomKey):
    chex.assert_shape(previous_samples, (num_batch, num_dim))
    normal_key, acceptance_key = jax.random.split(curr_key)
    normal_deltas = proposal_scale * jax.random.normal(
        key=normal_key, shape=(num_batch, num_dim))
    exponential_rvs = jax.random.exponential(key=acceptance_key,
                                             shape=(num_batch,))
    proposed_samples = previous_samples + normal_deltas
    chex.assert_shape(proposed_samples, (num_batch, num_dim))
    log_density_proposed = log_density_by_step(temp_step, proposed_samples)
    log_density_previous = log_density_by_step(temp_step, previous_samples)
    delta_log_prob = log_density_proposed - log_density_previous
    chex.assert_shape(delta_log_prob, (num_batch,))
    is_accepted = jnp.greater(delta_log_prob, -1.*exponential_rvs)
    chex.assert_shape(is_accepted, (num_batch,))
    step_acceptance_rate = jnp.mean(is_accepted * 1.)
    samples_next = jnp.where(is_accepted[:, None], proposed_samples,
                             previous_samples)
    chex.assert_shape(samples_next, (num_batch, num_dim))
    return samples_next, step_acceptance_rate

  keys = jax.random.split(key, num_mh_steps)
  samples_out, acceptance_rates = jax.lax.scan(rwm_step,
                                               samples_in,
                                               keys)
  acceptance_rate = jnp.mean(acceptance_rates)
  chex.assert_equal_shape((samples_out, samples_in))
  chex.assert_rank(acceptance_rate, 0)
  return samples_out, acceptance_rate


def momentum_step(samples_in: Array,
                  momentum_in: Array,
                  step_coefficient: Array,
                  epsilon: Array,
                  grad_log_density) -> Array:
  """A momentum update with variable momentum step_coefficient.

  Args:
    samples_in: (num_batch, num_dim)
    momentum_in: (num_batch, num_dim)
    step_coefficient: A Scalar which is typically either 0.5 (half step) or 1.0
    epsilon: A Scalar representing the constant step size.
    grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
  Returns:
    momentum_out (num_batch, num_dim)
  """
  chex.assert_rank((samples_in, step_coefficient, epsilon), (2, 0, 0))
  chex.assert_equal_shape((samples_in, momentum_in))
  momentum_out = momentum_in + step_coefficient * epsilon * grad_log_density(
      samples_in)
  chex.assert_equal_shape((momentum_in, momentum_out))
  return momentum_out


def leapfrog_step(samples_in: Array,
                  momentum_in: Array,
                  step_coefficient: Array,
                  epsilon: Array,
                  grad_log_density) -> Tuple[Array, Array]:
  """A step of the Leapfrog iteration with variable momentum step_coefficient.

  Args:
    samples_in: (num_batch, num_dim)
    momentum_in: (num_batch, num_dim)
    step_coefficient: A Scalar which is typically either 0.5 (half step) or 1.0
    epsilon: A Scalar representing the constant step size.
    grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
  Returns:
    samples_out: (num_batch, num_dim)
    momentum_out (num_batch, num_dim)
  """
  chex.assert_rank((samples_in, step_coefficient, epsilon), (2, 0, 0))
  chex.assert_equal_shape((samples_in, momentum_in))
  samples_out = samples_in + epsilon * momentum_in
  momentum_out = momentum_step(samples_out, momentum_in, step_coefficient,
                               epsilon, grad_log_density)
  chex.assert_equal_shape((samples_in, samples_out))
  return samples_out, momentum_out


def hmc_step(samples_in: Array,
             key: RandomKey,
             epsilon: Array,
             log_density,
             grad_log_density,
             num_leapfrog_iters: int) -> Tuple[Array, Array]:
  """A single step of Hamiltonian Monte Carlo.

  Args:
    samples_in: (num_batch, num_dim)
    key: A Jax random key.
    epsilon: A Scalar representing the constant step size.
    log_density: (num_batch, num_dim) -> (num_batch,)
    grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
    num_leapfrog_iters: Number of leapfrog iterations.
  Returns:
    samples_out: (num_batch, num_dim)
  """
  chex.assert_rank((samples_in, epsilon), (2, 0))
  num_batch = np.shape(samples_in)[0]
  samples_state = samples_in
  momentum_key, acceptance_key = jax.random.split(key)
  initial_momentum = jax.random.normal(momentum_key, np.shape(samples_in))
  # A half momentum step.
  momentum_state = momentum_step(samples_state, initial_momentum,
                                 step_coefficient=0.5,
                                 epsilon=epsilon,
                                 grad_log_density=grad_log_density)
  def scan_step(passed_state, unused_input):
    pos, mom = passed_state
    new_pos, new_mom = leapfrog_step(pos, mom, step_coefficient=1.0,
                                     epsilon=epsilon,
                                     grad_log_density=grad_log_density)
    return (new_pos, new_mom), None

  state_in = (samples_state, momentum_state)
  scan_length = num_leapfrog_iters - 1
  # (num_leapfrog_iters - 1) whole position and momentum steps.
  new_state, _ = jax.lax.scan(
      scan_step, state_in, [None] * scan_length, length=scan_length)
  samples_state, momentum_state = new_state

  # A whole position step and half momentum step.
  samples_state, momentum_state = leapfrog_step(
      samples_state,
      momentum_state,
      step_coefficient=0.5,
      epsilon=epsilon,
      grad_log_density=grad_log_density)

  # We don't negate the momentum here because it has no effect.
  # This would be required if momentum was used other than for just the energy.

  # Decide if we accept the proposed update using Metropolis correction.
  def get_combined_log_densities(pos, mom):
    pos_log_densities = log_density(pos)
    mom_log_densities = -0.5 * jnp.sum(jnp.square(mom), axis=1)
    return pos_log_densities + mom_log_densities

  current_log_densities = get_combined_log_densities(samples_in,
                                                     initial_momentum)
  proposed_log_densities = get_combined_log_densities(samples_state,
                                                      momentum_state)

  exponential_rvs = jax.random.exponential(key=acceptance_key,
                                           shape=(num_batch,))

  delta_log_prob = proposed_log_densities - current_log_densities
  chex.assert_shape(delta_log_prob, (num_batch,))
  is_accepted = jnp.greater(delta_log_prob, -1.*exponential_rvs)
  chex.assert_shape(is_accepted, (num_batch,))
  step_acceptance_rate = jnp.mean(is_accepted * 1.)
  samples_next = jnp.where(is_accepted[:, None], samples_state,
                           samples_in)

  return samples_next, step_acceptance_rate


def hmc(samples_in: Array,
        key: RandomKey,
        epsilon: Array,
        log_density,
        grad_log_density,
        num_leapfrog_iters: int,
        num_hmc_iters: int) -> Tuple[Array, Array]:
  """Hamiltonian Monte Carlo as described in Neal 2011.

  Args:
    samples_in: (num_batch, num_dim)
    key: A Jax random key.
    epsilon: A Scalar representing the constant step size.
    log_density: (num_batch, num_dim) -> (num_batch,)
    grad_log_density: (num_batch, num_dim) -> (num_batch, num_dim)
    num_leapfrog_iters: Number of leapfrog iterations.
    num_hmc_iters: Number of steps of Hamiltonian Monte Carlo.
  Returns:
    samples_out: (num_batch, num_dim)
  """
  step_keys = jax.random.split(key, num_hmc_iters)

  def short_hmc_step(loc_samples, loc_key):
    return hmc_step(loc_samples,
                    loc_key,
                    epsilon=epsilon,
                    log_density=log_density,
                    grad_log_density=grad_log_density,
                    num_leapfrog_iters=num_leapfrog_iters)

  samples_final, acceptance_rates = jax.lax.scan(short_hmc_step, samples_in,
                                                 step_keys)

  return samples_final, np.mean(acceptance_rates)


def hmc_wrapped(tfp_samples_in: Array,
                key: RandomKey,
                epsilon: Array,
                log_density_by_step: LogDensityByStep,
                temp_step: int,
                num_leapfrog_iters: int,
                num_hmc_iters: int
                ) -> Tuple[Array, Array]:
  """A wrapper for HMC that deals with all the interfacing with the codebase.

  Args:
    tfp_samples_in: (0, num_batch, num_dim)
    key: A Jax random key.
    epsilon: Scalar step size.
    log_density_by_step: Density at a given temperature.
    temp_step: Specifies the current temperature.
    num_leapfrog_iters: Number of leapfrog iterations.
    num_hmc_iters: Number of Hamiltonian Monte Carlo iterations.
  Returns:
    tfp_samples_out: (0, num_batch, num_dim)
  """
  samples_in = tfp_samples_in[0, :, :]
  log_density = lambda x: log_density_by_step(temp_step, x)
  unbatched_log_density = lambda x: log_density(x[None, :])[0]
  grad_log_density = jax.vmap(jax.grad(unbatched_log_density))
  samples_out, acceptance = hmc(
      samples_in,
      key=key,
      epsilon=epsilon,
      log_density=log_density,
      grad_log_density=grad_log_density,
      num_leapfrog_iters=num_leapfrog_iters,
      num_hmc_iters=num_hmc_iters)
  tfp_samples_out = samples_out[None, :, :]
  return tfp_samples_out, acceptance


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
    if hasattr(config, 'rwm_step_config'):
      self._rwm_step_size = InterpolatedStepSize(
          config.rwm_step_config,
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
    if self._config.rwm_steps_per_iter != 0:
      subkey, key = jax.random.split(key)
      samples, rwm_acc = random_walk_metropolis_tfp_compat(
          samples, self._rwm_step_size(step), self._density_by_step,
          step, self._config.rwm_steps_per_iter, subkey)
    else:
      rwm_acc = 1.
    if self._config.hmc_steps_per_iter != 0:
      if 'use_jax_hmc' in self._config and self._config.use_jax_hmc:
        samples, hmc_acc = hmc_wrapped(samples, key, self._hmc_step_size(step),
                                       self._density_by_step, step,
                                       self._config.hmc_num_leapfrog_steps,
                                       self._config.hmc_steps_per_iter)
      else:
        samples, hmc_acc = self._add_markov_kernel(
            samples, key, step, self._config.hmc_steps_per_iter,
            hmc_tfp_kernel, shared_trace_fn)
    else:
      hmc_acc = 1.

    acceptance_tuple = (nuts_acc, hmc_acc, rwm_acc)
    samples_out = samples[0, :, :]  # Remove axis to match format in this code.
    return samples_out, acceptance_tuple
