# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

import unittest
import setuptools

REQUIRED_PACKAGES = [
    'absl-py>=0.12.0',
    'chex>=0.0.7',
    'dm-haiku>=0.0.4',
    'jax>=0.2.16',
    'jaxlib>=0.1.68',
    'matplotlib>=3.4.2',
    'ml-collections>=0.1.0',
    'numpy>=1.19.5',
    'optax>=0.0.8',
    'pytest>=6.2.4',
    'scipy>=1.7.0',
    'tensorflow>=2.5.0',
    'tensorflow_probability>=0.13.0',
    'tensorflow_datasets>=4.3.0',
]


def aft_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover(
      'annealed_flow_transport', pattern='*_test.py')
  return test_suite

setuptools.setup(
    name='annealed_flow_transport',
    version='1.0',
    description='Implementation of Annealed Flow Transport Monte Carlo',
    url='https://github.com/deepmind/annealed_flow_transport',
    author='DeepMind',
    author_email='alexmatthews@google.com',
    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.aft_test_suite',
)
