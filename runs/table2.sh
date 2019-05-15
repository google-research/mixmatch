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

echo "# MixMatch for STL10 [default args --wd=0.02]"
common_args='--train_dir experiments/stl --beta=0.75 --w_match=50 --scales=4'
for seed in 1 2 3 4 5; do
    echo "python mixmatch.py --dataset=stl10.${seed}@1000-1 $common_args"
done
echo "python mixmatch.py --dataset=stl10.1@5000-1 $common_args"

