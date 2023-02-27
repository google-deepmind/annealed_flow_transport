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
from typing import Any, Callable, Mapping, NamedTuple, Tuple

import chex
import ml_collections
import numpy as np
import optax

VaeBatch = Mapping[str, np.ndarray]

ConfigDict = ml_collections.ConfigDict
Array = Any
Samples = chex.ArrayTree
SampleShape = Any
LogDensityByStep = Any
RandomKey = Array
AcceptanceTuple = Tuple[Array, Array, Array]
MarkovKernelApply = Callable[[int, RandomKey, Samples],
                             Tuple[Samples, AcceptanceTuple]]
OptState = optax.OptState
UpdateFn = optax.TransformUpdateFn
FlowParams = Any
FlowApply = Callable[[FlowParams, Samples], Tuple[Samples, Array]]
LogDensityNoStep = Callable[[Samples], Array]
InitialSampler = Callable[[RandomKey, int, Tuple[int]], Samples]
FreeEnergyAndGrad = Callable[[FlowParams, Array, Array, int], Tuple[Array,
                                                                    Array]]
FreeEnergyEval = Callable[[FlowParams, Array, Array, int], Array]
MNIST_IMAGE_SHAPE = (28, 28, 1)


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
  test_samples: Samples
  test_log_weights: Array
  log_normalizer_estimate: Array
  delta_time: float
  initial_time_diff: float


class ParticleState(NamedTuple):
  samples: Samples
  log_weights: Array
  log_normalizer_estimate: Array


class VAEResult(NamedTuple):
  sample_image: Array
  reconst_sample: Array
  latent_mean: Array
  latent_std: Array
  logits: Array


ParticlePropose = Callable[[RandomKey], ParticleState]
