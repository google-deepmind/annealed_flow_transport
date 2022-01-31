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

"""Evaluation code for launching PIMH and final target density MCMC."""


from absl import logging
from annealed_flow_transport import craft
from annealed_flow_transport import densities
from annealed_flow_transport import expectations
from annealed_flow_transport import flow_transport
from annealed_flow_transport import flows
from annealed_flow_transport import markov_kernel
from annealed_flow_transport import mcmc
from annealed_flow_transport import pimh
from annealed_flow_transport import samplers
from annealed_flow_transport import serialize
from annealed_flow_transport import smc
from annealed_flow_transport import vi
import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import jax.numpy as jnp

# Type defs.
Array = jnp.ndarray
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityByStep = tp.LogDensityByStep
RandomKey = tp.RandomKey
AcceptanceTuple = tp.AcceptanceTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
FreeEnergyEval = tp.FreeEnergyEval
MarkovKernelApply = tp.MarkovKernelApply
SamplesTuple = tp.SamplesTuple
LogWeightsTuple = tp.LogWeightsTuple
VfesTuple = tp.VfesTuple
InitialSampler = tp.InitialSampler
LogDensityNoStep = tp.LogDensityNoStep
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState
ParticlePropose = tp.ParticlePropose


class OnlineMovingAverage():
  """Numerically stable implementation of a moving average."""

  def __init__(self, label: str):
    self.label = label
    self._num_vals = 0

  # pytype: disable=attribute-error
  def update(self, val):
    self._num_vals += 1
    if self._num_vals == 1:
      self._average = val
    else:
      delta = (val - self._average)
      self._average = self._average + delta/self._num_vals

  def get_value(self):
    return self._average
  # pytype: enable=attribute-error


class ExpectationLogger(object):
  """Compute and log expectations based on particles."""

  def __init__(self, expectation_config, num_dim: int):
    self._expectation_config = expectation_config
    self._step = 0

    self._expectations = []
    for config in self._expectation_config.configurations:
      exp_class = getattr(expectations, config.name)
      self._expectations.append(jax.jit(exp_class(config, num_dim)))

    names = []
    self._labels = []
    for config in self._expectation_config.configurations:
      name = config.name
      index = len([elem for elem in names if elem == name])
      label = name+'_'+str(index)
      names.append(name)
      self._labels.append(label)

    self._averages = []
    for config, label in zip(self._expectation_config.configurations,
                             self._labels):
      self._averages.append(OnlineMovingAverage(label=label))

  def record_expectations(self, particle_state: ParticleState):
    """Record expectations based on particle state."""
    for index, expectation in enumerate(self._expectations):
      expectation_val = expectation(particle_state.samples,
                                    particle_state.log_weights)
      average = self._averages[index]
      average.update(expectation_val)

    if self._step % self._expectation_config.expectation_report_step == 0:
      logging.info('Step %05d :', self._step)
      msg = ''
      for average in self._averages:
        msg += average.label + ' '
        msg += str(average.get_value()) + ', '
      logging.info(msg)
    self._step += 1


def is_flow_algorithm(algo_name):
  return algo_name in ('craft', 'vi')


def is_annealing_markov_kernel_algorithm(algo_name):
  return algo_name in ('smc', 'craft')


def get_particle_propose(config) -> ParticlePropose:
  """Get a function that proposes particles and log normalizer."""
  log_density_initial = getattr(densities, config.initial_config.density)(
      config.initial_config, config.sample_shape[0])
  log_density_final = getattr(densities, config.final_config.density)(
      config.final_config, config.sample_shape[0])
  initial_sampler = getattr(samplers,
                            config.initial_sampler_config.initial_sampler)(
                                config.initial_sampler_config)

  if is_annealing_markov_kernel_algorithm(config.algo):
    density_by_step = flow_transport.GeometricAnnealingSchedule(
        log_density_initial, log_density_final, config.num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
        config.mcmc_config, density_by_step, config.num_temps)

  if is_flow_algorithm(config.algo):
    def flow_func(x):
      flow = getattr(flows, config.flow_config.type)(config.flow_config)
      return jax.vmap(flow)(x)
    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_params = serialize.load_checkpoint(config.params_filename)

  if config.algo == 'smc':
    @jax.jit
    def particle_propose(loc_key: RandomKey):
      return smc.fast_outer_loop_smc(density_by_step,
                                     initial_sampler,
                                     markov_kernel_by_step,
                                     loc_key,
                                     config)
  elif config.algo == 'craft':
    @jax.jit
    def particle_propose(loc_key: RandomKey):
      return craft.craft_evaluation_loop(loc_key,
                                         flow_params,
                                         flow_forward_fn.apply,
                                         markov_kernel_by_step,
                                         initial_sampler,
                                         density_by_step,
                                         config)
  elif config.algo == 'vi':
    @jax.jit
    def particle_propose(loc_key: RandomKey):
      return vi.vfe_naive_importance(initial_sampler,
                                     log_density_initial,
                                     log_density_final,
                                     flow_forward_fn.apply,
                                     flow_params,
                                     loc_key,
                                     config)
  else:
    raise NotImplementedError

  return particle_propose


def get_expectation_logger(expectation_config,
                           num_dim: int) -> ExpectationLogger:
  return ExpectationLogger(expectation_config, num_dim)


def run_experiment(config):
  """Run a SMC flow experiment.

  Args:
    config: experiment configuration.
  Returns:
    An AlgoResultsTuple containing the experiment results.
  """
  random_key = jax.random.PRNGKey(config.evaluation_seed)

  expectation_logger = get_expectation_logger(config.expectation_config,
                                              config.sample_shape[0])
  if config.evaluation_algo == 'pimh':
    particle_propose = get_particle_propose(config)

    logging.info('Draw initial samples redundantly for accurate timing...')
    particle_propose(random_key)
    logging.info('Starting PIMH algorithm.')
    pimh.particle_metropolis_loop(random_key,
                                  particle_propose,
                                  config.num_evaluation_samples,
                                  expectation_logger.record_expectations)
  elif config.evaluation_algo == 'mcmc_final':
    mcmc.outer_loop_mcmc(random_key,
                         config.num_evaluation_samples,
                         expectation_logger.record_expectations,
                         config)
  else:
    raise NotImplementedError
