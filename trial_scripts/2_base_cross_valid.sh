#!/bin/bash

scripts=(
  # "python src/train_kfold.py --window_size 24 --mode 'base'"
  # "python src/train_kfold.py --window_size 28 --mode 'base'"
  # "python src/train_kfold.py --window_size 32 --mode 'base'"
  # "python src/train_kfold.py --window_size 36 --mode 'base'"
  # "python src/train_kfold.py --window_size 40 --mode 'base'"
  # "python src/train_kfold.py --window_size 44 --mode 'base'"
  # "python src/train_kfold.py --window_size 48 --mode 'base'"
  # "python src/train_kfold.py --window_size 52 --mode 'base'"
  # "python src/train_kfold.py --window_size 56 --mode 'base'"
  # "python src/train_kfold.py --window_size 60 --mode 'base'"
  # "python src/train_kfold.py --window_size 64 --mode 'base'"
  # "python src/train_kfold.py --window_size 68 --mode 'base'"
  # "python src/train_kfold.py --window_size 72 --mode 'base'"
  # "python src/train_kfold.py --window_size 76 --mode 'base'"
  # "python src/train_kfold.py --window_size 80 --mode 'base'"
  # "python src/train_kfold.py --window_size 84 --mode 'base'"
  "python src/train_kfold.py --window_size 88 --mode 'base'"
  "python src/train_kfold.py --window_size 92 --mode 'base'"
  "python src/train_kfold.py --window_size 96 --mode 'base'"
  "python src/train_kfold.py --window_size 100 --mode 'base' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now