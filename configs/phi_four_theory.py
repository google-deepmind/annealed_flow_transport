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

"""An experiment setup for the phi^four theory experiment."""
import ml_collections
ConfigDict = ml_collections.ConfigDict


def get_config():
  """Returns a standard normal experiment config as ConfigDict."""
  config = ConfigDict()

  config.seed = 1
  config.evaluation_seed = 1
  config.batch_size = 2000
  config.estimation_batch_size = 2000
  config.sample_shape = (14*14,)
  config.report_step = 1
  config.vi_report_step = 100
  config.num_layers = 1
  config.step_logging = True
  config.num_temps = 1001
  config.resample_threshold = 0.3
  config.write_samples = True
  config.stopping_criterion = 'time'
  config.use_resampling = True
  config.use_markov = True
  config.algo = 'craft'
  config.evaluation_algo = 'pimh'
  config.num_evaluation_samples = 1000
  config.optim_markov = False
  config.craft_num_iters = 1000
  config.craft_batch_size = 2000
  config.vi_iters = 30000
  config.checkpoint_interval = 200000
  config.vi_estimator = 'importance'

  optimization_config = ConfigDict()
  optimization_config.free_energy_iters = 100
  optimization_config.aft_step_size = 1e-3
  optimization_config.craft_step_size = 1e-3
  optimization_config.vi_step_size = 1e-4
  config.optimization_config = optimization_config

  initial_config = ConfigDict()
  initial_config.density = 'MultivariateNormalDistribution'
  initial_config.shared_mean = 0.
  initial_config.diagonal_cov = 1.
  config.initial_config = initial_config

  final_config = ConfigDict()
  final_config.density = 'PhiFourTheory'
  final_config.bare_coupling = 5.1
  final_config.mass_squared = -4.75
  config.final_config = final_config

  flow_config = ConfigDict()
  flow_config.type = 'ConvAffineCouplingStack'
  flow_config.conv_kernel_shape = [3, 3]
  flow_config.conv_num_middle_layers = 1
  flow_config.conv_num_middle_channels = 10
  flow_config.num_coupling_layers = 2
  flow_config.is_torus = True
  flow_config.identity_init = True
  flow_config.num_elem = config.sample_shape[0]
  config.flow_config = flow_config
  initial_sampler_config = ConfigDict()
  initial_sampler_config.initial_sampler = 'MultivariateNormalDistribution'
  config.initial_sampler_config = initial_sampler_config

  mcmc_config = ConfigDict()
  hmc_step_config = ConfigDict()

  # Parameters for 14 x 14 m^2 = -4.75
  hmc_step_config.step_times = [0., 0.3, 1.]
  hmc_step_config.step_sizes = [0.3, 0.15, 0.1]

  nuts_step_config = ConfigDict()
  nuts_step_config.step_times = [0., 0.25, 0.5, 1.]
  nuts_step_config.step_sizes = [0.7, 0.7, 0.5, 0.5]

  mcmc_config.hmc_step_config = hmc_step_config
  mcmc_config.slice_step_config = hmc_step_config
  mcmc_config.nuts_step_config = nuts_step_config
  mcmc_config.hmc_steps_per_iter = 10
  mcmc_config.rwm_steps_per_iter = 0
  mcmc_config.hmc_num_leapfrog_steps = 10

  mcmc_config.slice_steps_per_iter = 0
  mcmc_config.nuts_steps_per_iter = 0
  mcmc_config.slice_max_doublings = 5
  mcmc_config.nuts_max_tree_depth = 4
  mcmc_config.iters = 1
  config.mcmc_config = mcmc_config

  config.save_params = True
  config.params_filename = ''

  expectation_config = ConfigDict()
  expectation_config.expectation_report_step = 50
  two_point_config = ConfigDict()
  two_point_config.name = 'TwoPointSusceptibility'
  ising_config = ConfigDict()
  ising_config.name = 'IsingEnergyDensity'
  one_site_config = ConfigDict()
  one_site_config.name = 'SingleComponentMean'
  one_site_config.component_index = 0
  expectation_config.configurations = [two_point_config,
                                       ising_config]
  config.expectation_config = expectation_config

  return config
