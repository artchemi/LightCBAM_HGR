#!/bin/bash

scripts=(
  "python src/train_kfold_CAM.py --window_size 24 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 28 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 32 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 36 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 40 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 44 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 48 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 52 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 56 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 60 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 64 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 68 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 72 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 76 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 80 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 84 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 88 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 92 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 96 --channels 'reduced' "
  "python src/train_kfold_CAM.py --window_size 100 --channels 'reduced' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now