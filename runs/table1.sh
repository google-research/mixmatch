#!/usr/bin/env bash

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

echo "# MixMatch on large models (26M parameters) for CIFAR10 and CIFAR100"
common_args='--train_dir experiments/large --wd=0.04 --beta=0.75 --filters=135'
for seed in 1 2 3 4 5; do
    echo "python mixmatch.py --dataset=cifar10.${seed}@4000-1 --w_match=75 $common_args"
    echo "python mixmatch.py --dataset=cifar100.${seed}@10000-1 --w_match=150 $common_args"
done

