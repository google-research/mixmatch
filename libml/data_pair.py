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

import itertools

from absl import flags
from libml.data import DataSet, augment_cifar10, augment_svhn, augment_stl10
import tensorflow as tf

flags.DEFINE_integer('nu', 2, 'Number of augmentations for class-consistency.')
FLAGS = flags.FLAGS


def stack_augment(augment):
    def func(x):
        xl = [augment(x) for _ in range(FLAGS.nu)]

        return dict(image=tf.stack([x['image'] for x in xl]),
                    label=tf.stack([x['label'] for x in xl]))

    return func


DATASETS = {}
DATASETS.update([DataSet.creator('cifar10', seed, label, valid, [augment_cifar10, stack_augment(augment_cifar10)])
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])
DATASETS.update(
    [DataSet.creator('cifar100', seed, label, valid, [augment_cifar10, stack_augment(augment_cifar10)], nclass=100)
     for seed, label, valid in
     itertools.product(range(6), [10000], [1, 5000])])
DATASETS.update([DataSet.creator('stl10', seed, label, valid, [augment_stl10, stack_augment(augment_stl10)], height=96,
                                 width=96, do_memoize=False)
                 for seed, label, valid in
                 itertools.product(range(6), [1000, 5000], [1, 500])])
DATASETS.update([DataSet.creator('svhn', seed, label, valid, [augment_svhn, stack_augment(augment_svhn)],
                                 do_memoize=False)
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])
DATASETS.update([DataSet.creator('svhn_noextra', seed, label, valid, [augment_svhn, stack_augment(augment_svhn)],
                                 do_memoize=False)
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000, 8000], [1, 5000])])
