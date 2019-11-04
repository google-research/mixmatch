# MixMatch - A Holistic Approach to Semi-Supervised Learning

Code for the paper: "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.

This is not an officially supported Google product.

## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create semi-supervised subsets
for seed in 1 2 3 4 5; do
    for size in 250 500 1000 2000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=10000 $ML_DATA/SSL/cifar100 $ML_DATA/cifar100-train.tfrecord &
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

#### Install privacy datasets

```bash
CUDA_VISIBLE_DEVICES= ./privacy/scripts/create_datasets.py

for size in 27 38 77 156 355 671 867; do
CUDA_VISIBLE_DEVICES= ./privacy/scripts/create_split.py --size=$size $ML_DATA/SSL/svhn500 $ML_DATA/svhn500-train.tfrecord &
done; wait

for size in 96 185 353 719 1415 2631 3523; do
CUDA_VISIBLE_DEVICES= ./privacy/scripts/create_split.py --size=$size $ML_DATA/SSL/svhn300 $ML_DATA/svhn300-train.tfrecord &
done; wait

for size in 56 81 109 138 266 525 1059 2171 4029 5371; do
CUDA_VISIBLE_DEVICES= ./privacy/scripts/create_split.py --size=$size $ML_DATA/SSL/svhn200 $ML_DATA/svhn200-train.tfrecord &
done; wait

for size in 145 286 558 1082 2172 4078 5488; do
CUDA_VISIBLE_DEVICES= ./privacy/scripts/create_split.py --size=$size $ML_DATA/SSL/svhn200s150 $ML_DATA/svhn200s150-train.tfrecord &
done; wait
```


## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a mixmatch with 32 filters on cifar10 shuffled with `seed=3`, 250 labeled samples and 5000
validation samples:
```bash
CUDA_VISIBLE_DEVICES=0 python mixmatch.py --filters=32 --dataset=cifar10.3@250-5000 --w_match=75 --beta=0.75
```

Available labelled sizes are 250, 500, 1000, 2000, 4000.
For validation, available sizes are 1, 5000 (and 500 for STL10).
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).

### Valid dataset names
```bash
for dataset in cifar10 svhn svhn_noextra; do
for seed in 1 2 3 4 5; do
for valid in 1 5000; do
for size in 250 500 1000 2000 4000; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
for valid in 1 500; do
    echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"
```


## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir experiments/
```

## Checkpoint accuracy

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
./scripts/extract_accuracy.py experiments/compare/cifar10.3@250-5000/MixMatch_archresnet_batch64_beta0.75_ema0.999_filters32_lr0.002_nclass10_repeat4_scales3_w_match75.0_wd0.02
# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Reproducing tables from the paper

Check the contents of the `runs/*.sh` files, these will give you the commands (and the hyper-parameters) to reproduce the results from the paper.

## Citing this work

```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
