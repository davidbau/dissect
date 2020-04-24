#!/usr/bin/env bash
cd models
# The VGG-19 network is obtained by:
# 1. converting vgg_normalised.caffemodel to .t7 using loadcaffe
# 2. inserting a convolutional module at the beginning to preprocess the image
# 3. replacing zero-padding with reflection-padding
# The original vgg_normalised.caffemodel can be obtained with:
# "wget -c --no-check-certificate https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel"
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=108uza-dsmwvbW2zv-G73jtVcMU_2Nb7Y' -O vgg_normalised.pth
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1w9r1NoYnn7tql1VYG3qDUzkbIks24RBQ' -O decoder.pth
cd ..
