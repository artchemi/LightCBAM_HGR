#!/bin/bash

scripts=(
  "python src/train_kfold.py --window_size 60 --mode 'attention' "
  "python src/train_kfold.py --window_size 70 --mode 'attention' "
  "python src/train_kfold.py --window_size 80 --mode 'attention' "
  "python src/train_kfold.py --window_size 90 --mode 'attention' "
  "python src/train_kfold.py --window_size 100 --mode 'attention' "
  "python src/train_kfold.py --window_size 110 --mode 'attention' "
  "python src/train_kfold.py --window_size 120 --mode 'attention' "
  "python src/train_kfold.py --window_size 130 --mode 'attention' "
  "python src/train_kfold.py --window_size 140 --mode 'attention' "
  "python src/train_kfold.py --window_size 150 --mode 'attention' "
  "python src/train_kfold.py --window_size 160 --mode 'attention' "
  "python src/train_kfold.py --window_size 170 --mode 'attention' "
  "python src/train_kfold.py --window_size 180 --mode 'attention' "
  "python src/train_kfold.py --window_size 190 --mode 'attention' "
  "python src/train_kfold.py --window_size 200 --mode 'attention' "
  "python src/train_kfold.py --window_size 210 --mode 'attention' "
  "python src/train_kfold.py --window_size 220 --mode 'attention' "
  "python src/train_kfold.py --window_size 230 --mode 'attention' "
  "python src/train_kfold.py --window_size 240 --mode 'attention' "
  "python src/train_kfold.py --window_size 250 --mode 'attention' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now