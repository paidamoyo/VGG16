#!/bin/bash

for SEED in 1 2 3 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    for LR in 1 2 3 4
    do
        python model_unsuper.py $SEED $LR
    done
done