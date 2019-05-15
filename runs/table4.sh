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

echo "# Ablation study, we're using seed 3 (which we found to be the hardest one for CIFAR10)"
echo "# Using default options:  --wd=0.02 --filters=32"
for size in 250 4000; do
    common_args="--dataset=cifar10.3@${size}-1 --train_dir experiments/ablation --beta=0.75"
    echo "python ablation/ab_mixmatch.py $common_args  # Standard MixMatch"
    echo "python ablation/ab_mixmatch.py $common_args --nu=1  # No consistency"
    echo "python ablation/ab_mixmatch.py $common_args --T=1  # No sharpening"
    echo "python ablation/ab_mixmatch.py $common_args --use_ema_guess  # EMA guess"
    echo "python ablation/ab_mixmatch.py $common_args --mixmode=.  # No mixup"
    echo "python ablation/ab_mixmatch.py $common_args --mixmode=xx.  # Only mixup labeled"
    echo "python ablation/ab_mixmatch.py $common_args --mixmode=.yy  # Only mixup unlabeled"
    echo "python ablation/ab_mixmatch.py $common_args --mixmode=xx.yy  # Mixup labeled and unlabeled separately"
    echo "python ict.py $common_args --consistency_weight=1000  # ICT"
    echo
done

