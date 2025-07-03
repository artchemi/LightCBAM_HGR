#!/bin/bash

scripts=(
  # "python src/train_kfold.py --window_size 24 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 28 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 32 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 36 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 40 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 44 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 48 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 52 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 56 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 60 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 64 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 68 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 72 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 76 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 80 --mode 'reduced'"
  # "python src/train_kfold.py --window_size 84 --mode 'reduced'"
  "python src/train_kfold.py --window_size 88 --mode 'reduced'"
  "python src/train_kfold.py --window_size 92 --mode 'reduced'"
  "python src/train_kfold.py --window_size 96 --mode 'reduced'"
  "python src/train_kfold.py --window_size 100 --mode 'reduced' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now