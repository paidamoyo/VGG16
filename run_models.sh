#!/bin/bash

for LR in 1 2 3 4
do
    for BATCH in 12 24
    do
        for SEED in 3 4 5 6 7
        do
            python model_3.py $LR $BATCH $SEED
        done
    done
done