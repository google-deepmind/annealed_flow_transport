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

"""Code for exact sampling from initial distributions."""

from typing import Tuple

import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp

RandomKey = tp.RandomKey
Array = jnp.ndarray


class NormalDistribution(object):
  """A wrapper for the univariate normal sampler."""

  def __init__(self, config):
    self._config = config

  def __call__(self,
               key: RandomKey,
               num_samples: int,
               sample_shape: Tuple[int]) -> Array:
    batched_sample_shape = (num_samples,) + sample_shape
    return jax.random.normal(key,
                             shape=batched_sample_shape)


class MultivariateNormalDistribution(object):
  """A wrapper for the multivariate normal sampler."""

  def __init__(self, config):
    self._config = config

  def __call__(self, key: RandomKey, num_samples: int,
               sample_shape: Tuple[int]) -> Array:
    batched_sample_shape = (num_samples,) + sample_shape
    return jax.random.normal(key, shape=batched_sample_shape)
