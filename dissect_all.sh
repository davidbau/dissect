#!/bin/bash

QUANTILE=0.01
MINIOU=0.04
SEG=netpqc
DATASET=places

MODEL=vgg16
for LAYER in conv1_1 conv1_2 conv2_1 conv2_2 conv3_1 conv3_2 conv3_3 \
     conv4_1 conv4_2 conv4_3 conv5_1 conv5_2 conv5_3
do

python -m experiment.dissect_experiment \
    --quantile ${QUANTILE} --miniou ${MINIOU} \
    --model ${MODEL} --dataset ${DATASET} --seg ${SEG} --layer ${LAYER}

done

MODEL=resnet152
for LAYER in 0 4 5 6 7
do

python -m experiment.dissect_experiment \
    --quantile ${QUANTILE} --miniou ${MINIOU} \
    --model ${MODEL} --dataset ${DATASET} --seg ${SEG} --layer ${LAYER}

done

MODEL=alexnet
for LAYER in conv1 conv2 conv3 conv4 conv5
do

python -m experiment.dissect_experiment \
    --quantile ${QUANTILE} --miniou ${MINIOU} \
    --model ${MODEL} --dataset ${DATASET} --seg ${SEG} --layer ${LAYER}

done

MODEL=progan
for DATASET in kitchen church bedroom livingroom
do

for LAYER in layer1 layer2 layer3 layer4 layer5 layer6 layer7 \
    layer8 layer9 layer10 layer11 layer12 layer13 layer14
do

python -m experiment.dissect_experiment \
    --quantile ${QUANTILE} --miniou ${MINIOU} \
    --model ${MODEL} --dataset ${DATASET} --seg ${SEG} --layer ${LAYER}

done

done

