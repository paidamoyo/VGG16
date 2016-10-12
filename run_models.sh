#!/bin/sh

for SPLIT in 1 2 3 4 5 6 7
do
    for LR in 1 2 3
    do
        for SEED in 1 2 3 4 5
        do
            python model_3.py $SPLIT $LR $SEED
        done
    done
done