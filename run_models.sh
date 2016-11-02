#!/bin/bash

for NUM in {1..60}
do
    python model_unsuper.py $NUM
done