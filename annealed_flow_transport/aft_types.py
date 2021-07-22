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

"""Shared custom defined types used in more than one source file."""
from typing import Any, Callable, NamedTuple, Tuple

import jax.numpy as jnp
import ml_collections
import optax

ConfigDict = ml_collections.ConfigDict
Array = jnp.ndarray
LogDensityByStep = Callable[[int, Array], Array]
RandomKey = Array
AcceptanceTuple = Tuple[Array, Array]
MarkovKernelApply = Callable[[int, RandomKey, Array], Tuple[Array,
                                                            AcceptanceTuple]]
OptState = optax.OptState
UpdateFn = optax.GradientTransformation
FlowParams = Any
FlowApply = Callable[[FlowParams, Array], Tuple[Array, Array]]
LogDensityNoStep = Callable[[Array], Array]
InitialSampler = Callable[[RandomKey, int, Tuple[int]], Array]
FreeEnergyAndGrad = Callable[[FlowParams, Array, Array, int], Tuple[Array,
                                                                    Array]]
FreeEnergyEval = Callable[[FlowParams, Array, Array, int], Array]


class SamplesTuple(NamedTuple):
  train_samples: Array
  validation_samples: Array
  test_samples: Array


class LogWeightsTuple(NamedTuple):
  train_log_weights: Array
  validation_log_weights: Array
  test_log_weights: Array


class VfesTuple(NamedTuple):
  train_vfes: Array
  validation_vfes: Array


class AlgoResultsTuple(NamedTuple):
  test_samples: Array
  test_log_weights: Array
  log_normalizer_estimate: Array
  delta_time: float
  initial_time_diff: float
