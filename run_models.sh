#!/bin/bash

for SPLIT in 1 2 3 4
do
    for LR in 1 2 3
    do
        for SEED in 1 2 3
        do
            python model_3.py $SPLIT $LR $SEED
        done
    done
done