#!/bin/bash

for SIGMA in 1000 100 10 1 0.1 0.001 0.0001 0.00001
do
    for LR in 1 2 3 4
    do
        python model_unsuper.py $SIGMA $LR
    done
done