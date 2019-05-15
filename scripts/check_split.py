#!/usr/bin/env python

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

"""Script to measure the overlap between data splits.

There should not be any overlap unless the original dataset has duplicates.
"""

import hashlib
import itertools
import os
from absl import app
from absl import flags
from libml import data, utils
import tensorflow as tf
from tqdm import trange

flags.DEFINE_integer('batch', 1024, 'Batch size.')
flags.DEFINE_integer('samples', 1 << 20, 'Number of samples to load.')

FLAGS = flags.FLAGS

DATASETS = {}
DATASETS.update([data.DataSet.creator('cifar10', seed, label, valid, lambda x: x)
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000], [1, 5000])])
DATASETS.update([data.DataSet.creator('cifar100', seed, label, valid, lambda x: x)
                 for seed, label, valid in
                 itertools.product(range(6), [10000], [1, 5000])])
DATASETS.update([data.DataSet.creator('stl10', seed, label, valid, lambda x: x, height=96, width=96, do_memoize=False)
                 for seed, label, valid in itertools.product(range(6), [1000, 5000], [1, 500])])
DATASETS.update([data.DataSet.creator('svhn', seed, label, valid, lambda x: x, do_memoize=False)
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000], [1, 5000])])
DATASETS.update([data.DataSet.creator('svhn_noextra', seed, label, valid, lambda x: x, do_memoize=False)
                 for seed, label, valid in
                 itertools.product(range(6), [250, 500, 1000, 2000, 4000], [1, 5000])])


def to_byte(d: dict):
    return tf.to_int32(tf.round(127.5 * (d['image'] + 1)))


def collect_hashes(sess, group, data):
    data = data.map(to_byte).batch(FLAGS.batch).prefetch(1).make_one_shot_iterator().get_next()
    hashes = set()
    hasher = hashlib.sha512
    for _ in trange(0, FLAGS.samples, FLAGS.batch, desc='Building hashes for %s' % group, leave=False):
        try:
            batch = sess.run(data)
        except tf.errors.OutOfRangeError:
            break
        for img in batch:
            hashes.add(hasher(img).digest())
    return hashes


def main(argv):
    del argv
    utils.setup_tf()
    dataset = DATASETS[FLAGS.dataset]()
    with tf.Session(config=utils.get_config()) as sess:
        hashes = (collect_hashes(sess, 'labeled', dataset.eval_labeled),
                  collect_hashes(sess, 'unlabeled', dataset.eval_unlabeled),
                  collect_hashes(sess, 'validation', dataset.valid),
                  collect_hashes(sess, 'test', dataset.test))
    print('Overlap matrix (should be an almost perfect diagonal matrix with counts).')
    groups = 'labeled unlabeled validation test'.split()
    fmt = '%-10s %10s %10s %10s %10s'
    print(fmt % tuple([''] + groups))
    for p, x in enumerate(hashes):
        overlaps = [len(x & y) for y in hashes]
        print(fmt % tuple([groups[p]] + overlaps))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    app.run(main)
