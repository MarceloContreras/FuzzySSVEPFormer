#!/bin/bash

DATA_PATH="/home/marcelo/Documents/Tesis/FuzzySSVEPFormer/datasets/2015_Nakanishi_SSVEP_database"
PARAMS_PATH="/home/marcelo/Documents/Tesis/FuzzySSVEPFormer/datasets/Nakanishi.yaml"

declare -A EPOCHS
EPOCHS[ssvepformer]=100
EPOCHS[f1ssvepformer_C]=150
EPOCHS[f2ssvepformer_C]=150
EPOCHS[EEGNet]=600
EPOCHS[SSVEPNet]=500
EPOCHS[Deformer]=400

MODELS=("f1ssvepformer_C" "f2ssvepformer_C")
SIGNAL_SIZES=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")
SEEDS=(1 10 30 40 50)

declare -A EXTRA_ARGS
EXTRA_ARGS[SSVEPNet]="--no-compile"
EXTRA_ARGS[EEGNet]="--no-init_weights"
EXTRA_ARGS[Deformer]="--no-init_weights"

for model in "${MODELS[@]}"; do
    for signal_size in "${SIGNAL_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do

            epochs=${EPOCHS[$model]}
            extra_args=${EXTRA_ARGS[$model]}

            echo "Running $model | signal=${signal_size}s | seed=$seed | epochs=$epochs"

            python train.py \
                --model "$model" \
                --epochs "$epochs" \
                --signal_size "$signal_size" \
                --seed "$seed" \
                --data_path "$DATA_PATH" \
                --params_path "$PARAMS_PATH" \
                --dataset NAKANISHI \
                --save_model \
                $extra_args

        done
    done
done