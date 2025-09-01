#!/bin/bash

scripts=(
  "python src/train_kfold_CAM.py --window_size 60 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 70 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 80 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 90 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 100 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 110 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 120 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 130 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 140 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 150 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 160 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 170 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 180 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 190 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 200 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 210 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 220 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 230 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 240 --channels 'full' "
  "python src/train_kfold_CAM.py --window_size 250 --channels 'full' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now