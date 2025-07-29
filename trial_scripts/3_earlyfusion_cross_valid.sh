#!/bin/bash

scripts=(
  "python src/train_kfold_fusion.py --window_size 24 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 28 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 32 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 36 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 40 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 44 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 48 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 52 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 56 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 60 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 64 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 68 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 72 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 76 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 80 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 84 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 88 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 92 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 96 --mode 'early_fusion' "
  "python src/train_kfold_fusion.py --window_size 100 --mode 'early_fusion' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now