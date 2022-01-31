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

"""Training for all SMC and flow algorithms."""

from typing import Callable, Tuple
from annealed_flow_transport import aft
from annealed_flow_transport import craft
from annealed_flow_transport import densities
from annealed_flow_transport import flows
from annealed_flow_transport import samplers
from annealed_flow_transport import serialize
from annealed_flow_transport import smc
from annealed_flow_transport import snf
from annealed_flow_transport import vi
import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

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


def get_optimizer(initial_learning_rate: float,
                  boundaries_and_scales):
  """Get an optimizer possibly with learning rate schedule."""
  if boundaries_and_scales is None:
    return optax.adam(initial_learning_rate)
  else:
    schedule_fn = optax.piecewise_constant_schedule(
        initial_learning_rate,
        boundaries_and_scales[0])
    opt = optax.chain(optax.scale_by_adam(),
                      optax.scale_by_schedule(schedule_fn), optax.scale(-1.))
    return opt


def value_or_none(value: str,
                  config):
  if value in config:
    return config[value]
  else:
    return None


def prepare_outer_loop(initial_sampler: InitialSampler,
                       initial_log_density: Callable[[Array], Array],
                       final_log_density: Callable[[Array], Array],
                       flow_func: Callable[[Array], Tuple[Array, Array]],
                       config) -> AlgoResultsTuple:
  """Shared code outer loops then calls the outer loops themselves.

  Args:
    initial_sampler: Function for producing initial sample.
    initial_log_density: Function for evaluating initial log density.
    final_log_density: Function for evaluating final log density.
    flow_func: Flow function to pass to Haiku transform.
    config: experiment configuration.
  Returns:
    An AlgoResultsTuple containing the experiment results.

  """

  key = jax.random.PRNGKey(config.seed)

  flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
  key, subkey = jax.random.split(key)
  single_normal_sample = initial_sampler(subkey,
                                         config.batch_size,
                                         config.sample_shape)
  key, subkey = jax.random.split(key)
  flow_init_params = flow_forward_fn.init(subkey,
                                          single_normal_sample)

  if value_or_none('save_checkpoint', config):
    def save_checkpoint(params):
      return serialize.save_checkpoint(config.params_filename, params)
  else:
    save_checkpoint = None

  if config.algo == 'vi':
    # Add a save_checkpoint function here to enable saving final state.
    opt = get_optimizer(
        config.optimization_config.vi_step_size,
        None)
    opt_init_state = opt.init(flow_init_params)
    results = vi.outer_loop_vi(initial_sampler=initial_sampler,
                               opt_update=opt.update,
                               opt_init_state=opt_init_state,
                               flow_init_params=flow_init_params,
                               flow_apply=flow_forward_fn.apply,
                               key=key,
                               initial_log_density=initial_log_density,
                               final_log_density=final_log_density,
                               config=config,
                               save_checkpoint=save_checkpoint)
  elif config.algo == 'smc':
    results = smc.outer_loop_smc(initial_log_density=initial_log_density,
                                 final_log_density=final_log_density,
                                 initial_sampler=initial_sampler,
                                 key=key,
                                 config=config)
  elif config.algo == 'snf':
    opt = get_optimizer(
        config.optimization_config.snf_step_size,
        value_or_none('snf_boundaries_and_scales',
                      config.optimization_config))
    log_step_output = None
    results = snf.outer_loop_snf(flow_init_params, flow_forward_fn.apply,
                                 initial_log_density, final_log_density,
                                 initial_sampler, key, opt, config,
                                 log_step_output, save_checkpoint)
  elif config.algo == 'aft':
    opt = get_optimizer(
        config.optimization_config.aft_step_size,
        None)
    opt_init_state = opt.init(flow_init_params)
    # Add a log_step_output function here to enable non-trivial step logging.
    log_step_output = None
    results = aft.outer_loop_aft(opt_update=opt.update,
                                 opt_init_state=opt_init_state,
                                 flow_init_params=flow_init_params,
                                 flow_apply=flow_forward_fn.apply,
                                 initial_log_density=initial_log_density,
                                 final_log_density=final_log_density,
                                 initial_sampler=initial_sampler,
                                 key=key,
                                 config=config,
                                 log_step_output=log_step_output)
  elif config.algo == 'craft':
    opt = get_optimizer(
        config.optimization_config.craft_step_size,
        value_or_none('craft_boundaries_and_scales',
                      config.optimization_config))
    opt_init_state = opt.init(flow_init_params)
    log_step_output = None
    results = craft.outer_loop_craft(opt.update, opt_init_state,
                                     flow_init_params, flow_forward_fn.apply,
                                     initial_log_density, final_log_density,
                                     initial_sampler, key, config,
                                     log_step_output,
                                     save_checkpoint)
  else:
    raise NotImplementedError
  return results


def is_flow_algorithm(algo_name):
  return algo_name in ('aft', 'vi', 'craft', 'snf')


def run_experiment(config) -> AlgoResultsTuple:
  """Run a SMC flow experiment.

  Args:
    config: experiment configuration.
  Returns:
    An AlgoResultsTuple containing the experiment results.
  """
  log_density_initial = getattr(densities, config.initial_config.density)(
      config.initial_config, config.sample_shape[0])
  log_density_final = getattr(densities, config.final_config.density)(
      config.final_config, config.sample_shape[0])
  initial_sampler = getattr(samplers,
                            config.initial_sampler_config.initial_sampler)(
                                config.initial_sampler_config)

  def flow_func(x):
    if is_flow_algorithm(config.algo):
      flow = getattr(flows, config.flow_config.type)(config.flow_config)
      return jax.vmap(flow)(x)
    else:
      return None

  results = prepare_outer_loop(initial_sampler, log_density_initial,
                               log_density_final, flow_func, config)
  return results
