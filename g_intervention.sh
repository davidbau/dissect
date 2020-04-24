#!/bin/bash

DATASET=church
MODEL=progan
LAYER=layer4

python -m experiment.generator_int_experiment \
    --model ${MODEL} --dataset ${DATASET} --layer ${LAYER}

