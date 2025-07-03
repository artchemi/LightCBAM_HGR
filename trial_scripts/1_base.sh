#!/bin/bash

scripts=(
  "python src/train.py --window_size 24  "
  "python src/train.py --window_size 28  "
  "python src/train.py --window_size 32  "
  "python src/train.py --window_size 36  "
  "python src/train.py --window_size 40  "
  "python src/train.py --window_size 44  "
  "python src/train.py --window_size 48  "
  "python src/train.py --window_size 52  "
  "python src/train.py --window_size 56  "
  "python src/train.py --window_size 60  "
  "python src/train.py --window_size 64  "
  "python src/train.py --window_size 68  "
  "python src/train.py --window_size 72  "
  "python src/train.py --window_size 76  "
  "python src/train.py --window_size 80  "
  "python src/train.py --window_size 84  "
  "python src/train.py --window_size 88  "
  "python src/train.py --window_size 92  "
  "python src/train.py --window_size 96  "
  "python src/train.py --window_size 100  "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now