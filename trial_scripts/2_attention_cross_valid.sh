#!/bin/bash

scripts=(
  "python src/train_kfold.py --window_size 24 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 28 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 32 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 36 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 40 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 44 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 48 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 52 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 56 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 60 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 64 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 68 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 72 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 76 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 80 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 84 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 88 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 92 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 96 --mode 'attention' --channels 'reduced' "
  "python src/train_kfold.py --window_size 100 --mode 'attention' --channels 'reduced' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now