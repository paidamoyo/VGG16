#!/bin/bash

for BATCH in 16 32 64 128
do
    for LR in 1 2 3 4
    do
        python model_unsuper.py $BATCH $LR
    done
done