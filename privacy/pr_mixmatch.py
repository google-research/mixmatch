# Copyright 2019 Google LLC
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

"""Private version of MixMatch: we only changed the datasets.
"""
import os

from absl import app
from absl import flags
from libml import utils
from mixmatch import MixMatch
from privacy.lib.data_pair import DATASETS

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    assert FLAGS.nu == 2
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,

        beta=FLAGS.beta,
        w_match=FLAGS.w_match,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.04, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.75, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 75, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.002)
    FLAGS.set_default('train_kimg', 1 << 13)
    app.run(main)
