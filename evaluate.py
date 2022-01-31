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

"""Entry point for evaluation algorithms."""

from typing import Sequence

from absl import app
from absl import flags
from annealed_flow_transport import evaluation
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config',
                                './configs/single_normal.py',
                                'Evaluate configuration.')


def main(argv: Sequence[str]) -> None:
  config = FLAGS.config
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  evaluation.run_experiment(config)

if __name__ == '__main__':
  app.run(main)
