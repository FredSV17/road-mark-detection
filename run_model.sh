#!/bin/bash

# Loop from 0 to 1.0 in increments of 0.2
for i in 0 0.2 0.4 0.6 0.8 1.0
do
    echo "Running training with mosaic=$i"
    python -m model_training.train --path ./data --mosaic $i

    # Optional: check if the last command failed
    if [ $? -ne 0 ]; then
        echo "Training failed at mosaic=$i. Stopping."
        exit 1
    fi
done