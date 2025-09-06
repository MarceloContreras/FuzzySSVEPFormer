#!/bin/bash

noises=(1.0 10.0 100.0 1000.0 10000.0 100000.0)
time_windows=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 

for n in "${noises[@]}"; do
    poetry run python src/experiment_noise.py --noise $n --sub $1 --model ssvepformer --dataset NAKANISHI
    poetry run python src/experiment_noise.py --noise $n --sub $1 --model f1ssvepformer_C --dataset NAKANISHI
    poetry run python src/experiment_noise.py --noise $n --sub $1 --model f2ssvepformer_C --dataset NAKANISHI
done

for t in "${time_windows[@]}"; do
    poetry run python src/experiment_time.py --time $t --sub $1 --model ssvepformer --dataset NAKANISHI
    poetry run python src/experiment_time.py --time $t --sub $1 --model f1ssvepformer_C --dataset NAKANISHI
    poetry run python src/experiment_time.py --time $t --sub $1 --model f2ssvepformer_C --dataset NAKANISHI
done