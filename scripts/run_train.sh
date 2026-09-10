#!/bin/bash

SIGNAL_SIZES=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
SEEDS=(1 10 30 40 50)


for signal_size in "${SIGNAL_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do

        python thirdparty/Brain-computer-interfaces/trac_training_utec.py \
            --signal_size "$signal_size" 

    done
done
