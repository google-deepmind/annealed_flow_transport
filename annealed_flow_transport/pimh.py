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

"""Particle independent Metropolis-Hasting algorithm code."""
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp

Array = jnp.ndarray
ParticleState = tp.ParticleState
RandomKey = tp.RandomKey


def particle_metropolis_step(key: RandomKey,
                             current_particle_state: ParticleState,
                             proposed_particle_state) -> ParticleState:
  """A Particle independent Metropolis Hasting step.

  Accept the proposed particles with probability a(x', x_t)
  where x' is the proposed particle state, x_t is the current state and
  a(x', x_t) = min( 1, Z(x')/Z(x_t)) with Z being the normalizing constant
  estimate for that particle state.

  For numerical stability we transform the transformation to log space i.e.:
    u sim Uniform[0,1]
    Accept if a greater than u
  Becomes:
    log u sim -Exponential(1)
    Accept if log a greater than log u

  For more background see Andrieu, Doucet and Holenstein: 2010
    "Particle Markov chain Monte Carlo Methods" JRSS B.

  Args:
    key: A Jax random key.
    current_particle_state: Corresponds to x_t
    proposed_particle_state: Corresponds to x'
  Returns:
    next_particle_state: Results of the update step.
  """
  log_u = -1.*jax.random.exponential(key)
  log_a = proposed_particle_state.log_normalizer_estimate - current_particle_state.log_normalizer_estimate
  accept = log_a > log_u
  next_samples = jnp.where(accept,
                           proposed_particle_state.samples,
                           current_particle_state.samples)
  next_log_weights = jnp.where(accept,
                               proposed_particle_state.log_weights,
                               current_particle_state.log_weights)
  next_log_z = jnp.where(accept,
                         proposed_particle_state.log_normalizer_estimate,
                         current_particle_state.log_normalizer_estimate)
  next_particle_state = ParticleState(samples=next_samples,
                                      log_weights=next_log_weights,
                                      log_normalizer_estimate=next_log_z)
  return next_particle_state


def particle_metropolis_loop(key: RandomKey,
                             particle_propose,
                             num_samples: int,
                             record_expectations,
                             ):
  """Run a particle independent Metropolis-Hastings chain.

  Args:
    key: A Jax random key.
    particle_propose: Takes a RandomKey and returns a ParticleState.
    num_samples: Number of iterations to run for.
    record_expectations: Takes a ParticleState and logs required expectations.
  """
  subkey, key = jax.random.split(key)
  particle_state = particle_propose(subkey)
  record_expectations(particle_state)

  for unused_sample_index in range(num_samples):
    subkey, key = jax.random.split(key)
    proposed_particle_state = particle_propose(subkey)
    subkey, key = jax.random.split(key)
    particle_state = particle_metropolis_step(subkey,
                                              particle_state,
                                              proposed_particle_state)
    record_expectations(particle_state)


