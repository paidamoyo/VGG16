#!/bin/bash

for NUM in {1..10}
do
    i = $(($NUM + 100))
    j = $(($NUM + 200))
    python model_unsuper.py i
    python model_conv_auto.py j
done