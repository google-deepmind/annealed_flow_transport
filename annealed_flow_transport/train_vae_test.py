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

"""Tests for annealed_flow_transport.train_vae."""

from absl.testing import absltest
from absl.testing import parameterized
from annealed_flow_transport import train_vae


class TestEntryPoint(parameterized.TestCase):
  """Test that the main VAE training loop runs on a tiny example."""

  def test_entry_point(self):
    random_seed = 1
    batch_size = 2
    num_latents = 5
    step_size = 0.00005  #
    output_dir_stub = False
    train_iters = 7
    report_period = 3
    train_vae.train_vae(batch_size=batch_size,
                        num_latents=num_latents,
                        random_seed=random_seed,
                        step_size=step_size,
                        output_dir_stub=output_dir_stub,
                        train_iters=train_iters,
                        report_period=report_period)

if __name__ == '__main__':
  absltest.main()
