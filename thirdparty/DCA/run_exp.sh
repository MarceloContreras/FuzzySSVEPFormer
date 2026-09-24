#!/bin/bash

noises=(1.0 10.0 100.0 1000.0 10000.0 100000.0)
seeds=(0 1 2 3 4)
subjects=(5 6 7 8 9 10)

for sub in "${subjects[@]}"; do
    for n in "${noises[@]}"; do
        for seed in "${seeds[@]}"; do
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model ssvepformer --dataset NAKANISHI --seed $seed
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model f1ssvepformer_C --dataset NAKANISHI --seed $seed
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model f2ssvepformer_C --dataset NAKANISHI --seed $seed
        done
    done
done

subjects=(1 2 3 4 5 6 7 8 9 10)

for sub in "${subjects[@]}"; do
    for n in "${noises[@]}"; do
        for seed in "${seeds[@]}"; do
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model ssvepformer --dataset UTEC --seed $seed
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model f1ssvepformer_C --dataset UTEC --seed $seed
            poetry run python src/experiment_noise.py --noise $n --sub $sub --model f2ssvepformer_C --dataset UTEC --seed $seed
        done
    done
done

time_windows=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 

for sub in "${subjects[@]}"; do
    for t in "${time_windows[@]}"; do
        for seed in "${seeds[@]}"; do
            poetry run python src/experiment_time.py --time $t --sub $sub --model ssvepformer --dataset NAKANISHI --seed $seed
            poetry run python src/experiment_time.py --time $t --sub $sub --model f1ssvepformer_C --dataset NAKANISHI --seed $seed
            poetry run python src/experiment_time.py --time $t --sub $sub --model f2ssvepformer_C --dataset NAKANISHI --seed $seed
        done
    done
done

for sub in "${subjects[@]}"; do
    for t in "${time_windows[@]}"; do
        for seed in "${seeds[@]}"; do
            poetry run python src/experiment_time.py --time $t --sub $sub --model ssvepformer --dataset UTEC --seed $seed
            poetry run python src/experiment_time.py --time $t --sub $sub --model f1ssvepformer_C --dataset UTEC --seed $seed
            poetry run python src/experiment_time.py --time $t --sub $sub --model f2ssvepformer_C --dataset UTEC --seed $seed
        done
    done
done
