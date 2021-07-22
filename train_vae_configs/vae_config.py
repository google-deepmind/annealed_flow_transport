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

"""Configuration for train_vae.py."""

import ml_collections
ConfigDict = ml_collections.ConfigDict


def get_config():
  """Returns a train_vae config as ConfigDict."""
  config = ConfigDict()

  config.random_seed = 1
  config.batch_size = 100
  config.num_latents = 30  # Number of latents for VAE.
  config.step_size = 0.00005  # ADAM optimizer step size.
  # Base directory for output of results. If falsey don't store files.
  config.output_dir_stub = '/tmp/aft_vae/'
  config.train_iters = 500001
  config.report_period = 5000

  return config
