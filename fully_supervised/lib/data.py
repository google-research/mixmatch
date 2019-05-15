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

import os

from absl import flags
from libml import data, utils

FLAGS = flags.FLAGS


class DataSetFS(data.DataSet):
    @classmethod
    def creator(cls, name, train_files, test_files, valid, augment, parse_fn=data.default_parse, do_memoize=True,
                colors=3, nclass=10, height=32, width=32):
        train_files = [os.path.join(data.DATA_DIR, x) for x in train_files]
        test_files = [os.path.join(data.DATA_DIR, x) for x in test_files]
        fn = data.memoize if do_memoize else lambda x: x.repeat().shuffle(FLAGS.shuffle)

        def create():
            para = max(1, len(utils.get_available_gpus())) * FLAGS.para_augment
            train_labeled = parse_fn(data.dataset(train_files).skip(valid))
            if FLAGS.whiten:
                mean, std = data.compute_mean_std(train_labeled)
            else:
                mean, std = 0, 1

            return cls(name + '-' + str(valid),
                       train_labeled=fn(train_labeled).map(augment, para),
                       train_unlabeled=None,
                       eval_labeled=train_labeled.take(5000),  # No need to to eval on everything.
                       eval_unlabeled=None,
                       valid=parse_fn(data.dataset(train_files).take(valid)),
                       test=parse_fn(data.dataset(test_files)),
                       nclass=nclass, colors=colors, height=height, width=width, mean=mean, std=std)

        return name + '-' + str(valid), create


DATASETS = {}
DATASETS.update([DataSetFS.creator('cifar10', ['cifar10-train.tfrecord'], ['cifar10-test.tfrecord'], valid,
                                   data.augment_cifar10) for valid in [1, 5000]])
DATASETS.update([DataSetFS.creator('cifar100', ['cifar100-train.tfrecord'], ['cifar100-test.tfrecord'], valid,
                                   data.augment_cifar10, nclass=100) for valid in [1, 5000]])
DATASETS.update([DataSetFS.creator('stl10', [], [], valid, data.augment_stl10, height=96, width=96, do_memoize=False)
                 for valid in [1, 5000]])
DATASETS.update([DataSetFS.creator('svhn', ['svhn-train.tfrecord', 'svhn-extra.tfrecord'], ['svhn-test.tfrecord'],
                                   valid, data.augment_svhn, do_memoize=False) for valid in [1, 5000]])
DATASETS.update([DataSetFS.creator('svhn_noextra', ['svhn-train.tfrecord'], ['svhn-test.tfrecord'],
                                   valid, data.augment_svhn, do_memoize=False) for valid in [1, 5000]])
