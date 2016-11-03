#!/bin/bash

for NUM in {1..25}
do
    python model_unsuper.py ($NUM + 100)
    python model_conv_auto.py ($NUM + 200)
done