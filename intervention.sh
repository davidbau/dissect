#!/bin/bash

DATASET=places
MODEL=vgg16
LAYER=features.conv5_3

python -m experiment.intervention_experiment \
    --model ${MODEL} --dataset ${DATASET} --layer ${LAYER}

