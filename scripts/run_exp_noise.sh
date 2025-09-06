#!/bin/bash

for i in {1..10}; do
    python main.py --model=ssvepformer --sub "$i" --test_noise
    python main.py --model=f1ssvepformer_C --sub "$i" --test_noise
    python main.py --model=f2ssvepformer_C --sub "$i" --test_noise

    python main.py --model=ssvepformer --sub "$i" --test_time
    python main.py --model=f1ssvepformer_C --sub "$i" --test_time
    python main.py --model=f2ssvepformer_C --sub "$i" --test_time
done