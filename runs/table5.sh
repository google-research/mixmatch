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

echo "# CIFAR10 hyper-parameters for all methods under comparison"
for seed in 1 2 3 4 5; do
echo
for size in 250 500 1000 2000 4000; do
    common_args="--train_dir experiments/compare --dataset=cifar10.${seed}@${size}-1"
    echo "python ict.py $common_args --wd=0.02 --beta=0.75 --consistency_weight=1000"
    echo "python mean_teacher.py $common_args --wd=0.2 --smoothing=0.001 --consistency_weight=50"
    echo "python mixmatch.py $common_args --wd=0.02 --beta=0.75 --w_match=75"
    echo "python mixup.py $common_args --wd=0.02 --beta=0.75"
    echo "python pi_model.py $common_args --wd=0.02 --smoothing=0.001 --consistency_weight=20"
    echo "python pseudo_label.py $common_args --wd=0.02 --smoothing=0.01 --consistency_weight=1"
    echo "python vat.py $common_args --wd=0.02 --smoothing=0.01"
done
done
