#!/bin/bash

for NUM in {1..50}
do
    python model_unsuper.py $NUM
done