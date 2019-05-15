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
from libml.data import DataSet, augment_svhn
from libml.data_pair import stack_augment


DATASETS = {}
DATASETS.update(
    [DataSet.creator('svhn500', 0, label, valid, [augment_svhn, stack_augment(augment_svhn)], do_memoize=False)
     for label, valid in itertools.product([27, 38, 77, 156, 355, 671, 867], [1, 5000])])
DATASETS.update(
    [DataSet.creator('svhn300', 0, label, valid, [augment_svhn, stack_augment(augment_svhn)], do_memoize=False)
     for label, valid in itertools.product([96, 185, 353, 710, 1415, 2631, 3523], [1, 5000])])
DATASETS.update(
    [DataSet.creator('svhn200', 0, label, valid, [augment_svhn, stack_augment(augment_svhn)], do_memoize=False)
     for label, valid in itertools.product([56, 81, 109, 138, 266, 525, 1059, 2171, 4029, 5371], [1, 5000])])
DATASETS.update(
    [DataSet.creator('svhn200s150', 0, label, valid, [augment_svhn, stack_augment(augment_svhn)], do_memoize=False)
     for label, valid in itertools.product([145, 286, 558, 1082, 2172, 4078, 5488], [1, 5000])])
