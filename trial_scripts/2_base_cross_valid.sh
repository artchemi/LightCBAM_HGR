#!/bin/bash

scripts=(
  "python src/train_kfold.py --window_size 60 --mode 'base'"
  "python src/train_kfold.py --window_size 70 --mode 'base'"
  "python src/train_kfold.py --window_size 80 --mode 'base'"
  "python src/train_kfold.py --window_size 90 --mode 'base'"
  "python src/train_kfold.py --window_size 100 --mode 'base'"
  "python src/train_kfold.py --window_size 110 --mode 'base'"
  "python src/train_kfold.py --window_size 120 --mode 'base'"
  "python src/train_kfold.py --window_size 130 --mode 'base'"
  "python src/train_kfold.py --window_size 140 --mode 'base'"
  "python src/train_kfold.py --window_size 150 --mode 'base'"
  "python src/train_kfold.py --window_size 160 --mode 'base'"
  "python src/train_kfold.py --window_size 170 --mode 'base'"
  "python src/train_kfold.py --window_size 180 --mode 'base'"
  "python src/train_kfold.py --window_size 190 --mode 'base'"
  "python src/train_kfold.py --window_size 200 --mode 'base'"
  "python src/train_kfold.py --window_size 210 --mode 'base'"
  "python src/train_kfold.py --window_size 220 --mode 'base'"
  "python src/train_kfold.py --window_size 230 --mode 'base'"
  "python src/train_kfold.py --window_size 240 --mode 'base'"
  "python src/train_kfold.py --window_size 250 --mode 'base'"
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now