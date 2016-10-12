#!/bin/bash

for LR in 1 2 3 4
do
    for BATCH in 12 24 36
    do
        for SEED in 1 2 3 4 5 6 7 8 9 10
        do
            python model_3.py $LR $BATCH $SEED
        done
    done
done