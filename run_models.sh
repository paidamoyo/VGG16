#!/bin/bash

for SEED in 3
do
    for LR in 2
    do
        for BATCH in 16
        do
            python model_3.py $SEED $LR $BATCH
        done
    done
done