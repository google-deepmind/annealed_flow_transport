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

"""Code related to resampling of weighted samples."""
from typing import Tuple

import annealed_flow_transport.aft_types as tp
import chex
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
RandomKey = tp.RandomKey


def log_effective_sample_size(log_weights: Array) -> Array:
  """Numerically stable computation of log of effective sample size.

  ESS := (sum_i weight_i)^2 / (sum_i weight_i^2) and so working in terms of logs
  log ESS = 2 log sum_i (log exp log weight_i) - log sum_i (exp 2 log weight_i )

  Args:
    log_weights: Array of shape (num_batch). log of normalized weights.
  Returns:
    Scalar log ESS.
  """
  chex.assert_rank(log_weights, 1)
  first_term = 2.*jax.scipy.special.logsumexp(log_weights)
  second_term = jax.scipy.special.logsumexp(2.*log_weights)
  chex.assert_equal_shape([first_term, second_term])
  return first_term-second_term


def simple_resampling(key: RandomKey, log_weights: Array,
                      samples: Array) -> Tuple[Array, Array]:
  """Simple resampling of log_weights and samples pair.

  Randomly select possible samples with replacement proportionally to
  softmax(log_weights).

  Args:
    key: A Jax Random Key.
    log_weights: An array of size (num_batch,) containing the log weights.
    samples: An array of size (num_batch, num_dim) containing the samples.å
  Returns:
    New samples of shape (num_batch, num_dim) and weights of shape (num_batch,)
  """
  chex.assert_rank(log_weights, 1)
  num_batch = log_weights.shape[0]
  chex.assert_shape(samples, (num_batch, None))
  indices = jax.random.categorical(key, log_weights, shape=(samples.shape[0],))
  resamples = jnp.take(samples, indices, axis=0)
  log_weights_new = -np.log(log_weights.shape[0])*jnp.ones_like(log_weights)
  chex.assert_equal_shape([log_weights, log_weights_new])
  chex.assert_equal_shape([resamples, samples])
  return resamples, log_weights_new


def optionally_resample(key: RandomKey, log_weights: Array, samples: Array,
                        resample_threshold: Array) -> Tuple[Array, Array]:
  """Call simple_resampling on log_weights/samples if ESS is below threshold.

  The resample_threshold is interpretted as a fraction of the total number of
  samples. So for example a resample_threshold of 0.3 corresponds to an ESS of
  samples 0.3 * num_batch.

  Args:
    key: Jax Random Key.
    log_weights: Array of shape (num_batch,)
    samples: Array of shape (num_batch, num_dim)
    resample_threshold: scalar controlling fraction of total sample sized used.
  Returns:
    new samples of shape (num_batch, num_dim) and
  """
  # In the case where we don't resample we just return the current
  # samples and weights.
  # lamdba_no_resample will do that on the tuple given to jax.lax.cond below.
  lambda_no_resample = lambda x: (x[2], x[1])
  lambda_resample = lambda x: simple_resampling(*x)
  threshold_sample_size = log_weights.shape[0] * resample_threshold
  log_ess = log_effective_sample_size(log_weights)
  return jax.lax.cond(log_ess < np.log(threshold_sample_size), lambda_resample,
                      lambda_no_resample, (key, log_weights, samples))
