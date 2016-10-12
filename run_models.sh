#!/bin/bash

for SEED in 2 3 4 5 6 7 8 9 10
do
    for LR in 1 2 3
    do
        for BATCH in 16 32
        do
            python model_3.py $SEED $LR $BATCH
        done
    done
done