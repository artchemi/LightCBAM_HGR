#!/bin/bash

scripts=(
  "python src/subject_kfold.py --window_size 520 --mode 'base' --channels 'full' "
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

# sudo shutdown -h now