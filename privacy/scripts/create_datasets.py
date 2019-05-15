#!/usr/bin/env python3

# Copyright 2018 Google LLC
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

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import functools
import json
import os
import sys
import tempfile
from urllib import request

from libml.data import DATA_DIR
import numpy as np
import scipy.io
import tensorflow as tf
from tqdm import trange

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
}


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


# SVHN datasets for privacy are designed as follows:
# - First multiple teachers were trained with noisy SGD on train + extra.
# - The test set is then divided into two groups: (10000 samples for training, and 16032 for testing)
# - The teachers are used to label the training samples taken from the test set.
# - The 10,000 training samples taken from the test set are finally divided into the desired labeled and unlabeled
# groups.

def _load_private_svhn(name=None):
    splits = collections.OrderedDict()
    for split in ['test']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        if split == 'test':
            # svhnxxx, the xxx number is a setting for the privacy algorithm: the number of teachers that voted for
            # the same class.
            if name == 'svhn300':
                private_labels_file = 'data/privacy/svhn_noisy_labels_gnmax_t300_ts_200_n_40_david_convention.json'
            elif name == 'svhn500':
                private_labels_file = 'data/privacy/svhn_noisy_labels_gnmax_t500_ts_200_n_40_david_convention.json'
            elif name == 'svhn200':
                private_labels_file = 'data/privacy/svhn_noisy_labels_gnmax_t200_ts_200_n_40_david_convention.json'
            elif name == 'svhn200s150':
                private_labels_file = 'data/privacy/svhn_noisy_labels_gnmax_t200_ts_150_n_40_david_convention.json'
            dataset['labels'][:10000] = json.load(open(private_labels_file, 'r'))['label'][:10000]
            splits['train'] = dict(images=dataset['images'][:10000], labels=dataset['labels'][:10000])
            splits['test'] = dict(images=dataset['images'][10000:], labels=dataset['labels'][10000:])
        else:
            splits[split] = dataset
    return splits


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not os.path.exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)
    for filename, contents in files.items():
        with open(os.path.join(DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return os.path.exists(os.path.join(DATA_DIR, name, folder))


CONFIGS = dict(
    svhn200=dict(loader=functools.partial(_load_private_svhn, 'svhn200'),
                 checksums=dict(train=None, test=None, extra=None)),
    svhn200s150=dict(loader=functools.partial(_load_private_svhn, 'svhn200s150'),
                     checksums=dict(train=None, test=None, extra=None)),
    svhn300=dict(loader=functools.partial(_load_private_svhn, 'svhn300'),
                 checksums=dict(train=None, test=None, extra=None)),
    svhn500=dict(loader=functools.partial(_load_private_svhn, 'svhn500'),
                 checksums=dict(train=None, test=None, extra=None)),
)

if __name__ == '__main__':
    if len(sys.argv[1:]):
        subset = set(sys.argv[1:])
    else:
        subset = set(CONFIGS.keys())
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(DATA_DIR, file_and_data.filename)
                    open(path, "wb").write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))
