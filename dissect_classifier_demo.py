#!/usr/bin/env python
# Network Dissection (using new utility module)

# Set up experiment directory and settings

import torch, argparse, os, shutil, inspect, json, numpy, math
import netdissect
from netdissect.easydict import EasyDict
from netdissect import pbar, nethook, renormalize, parallelfolder, pidfile
from netdissect import upsample, tally, imgviz, imgsave, bargraph, show
from experiment import dissect_experiment as experiment
from netdissect import renormalize
from netdissect import imgviz

args = EasyDict(model='vgg16', dataset='places', seg='netpqc', layer='conv5_3', quantile=0.01)

# Just define a directory for saving some results
resdir = 'results/new-%s-%s-%s-%s-%s' % (
        args.model, args.dataset, args.seg, args.layer, int(args.quantile * 1000))
def resfile(f):
    return os.path.join(resdir, f)


# load classifier model and dataset
# Here I will use some utility functions that can download and create
# a pretrained model - you can just create the model as you like.
model = experiment.load_model(args, instrumented=False)
dataset = experiment.load_dataset(args)
sample_size = len(dataset)
percent_level = 1.0 - args.quantile

# This is just 'features.conv5_3'
layername = experiment.instrumented_layername(args)

print('Inspecting layer %s of model %s on %s' % (layername, args.model, args.dataset))


# Load segmenter, segment labels, classifier labels
classlabels = dataset.classes
segmodel, seglabels, segcatlabels = experiment.setting.load_segmenter(args.seg)
renorm = renormalize.renormalizer(dataset, target='zc')



# Collect quantile statistics
# Here i am using the newly checked-in dissect.py utility function
# to make a pass over the dataset and collect statistics over all
# units for the specified layer.
pbar.descnext('rq/topk')
from netdissect import dissect
topk, rq, run = dissect.acts_stats(model, dataset, layer=layername,
            k=200,
            batch_size=30,
            num_workers=30,
            cachedir=resdir)


# For unit visualizations....
# we can choose whatever output image size we like.  I set it to 100 pixels here.
# Here I am using the utility masked_images_for_topk to get five representative
# images for each unit.
pbar.descnext('unit_images')
iv = imgviz.ImageVisualizer((100, 100), source=dataset, quantiles=rq,
        level=rq.quantiles(percent_level))
unit_images = iv.masked_images_for_topk(
        run, dataset, topk, k=5, num_workers=30, pin_memory=True,
        cachefile=resfile('top5images.npz'))


# For text labels.....
# Here we use the new utility function to label according to
# the top matching labels within the top-20 images only.
level = rq.quantiles(percent_level)

# First compute all iou's between units and segment classes
all_iou = dissect.topk_label_stats_using_segmodel(dataset, segmodel,
        run, level, topk, k=20, downsample=4)

# Here we ignore the dummy label.  Any labels we aren'ts interested in
# can be ignored by zeroing their computed IOUs.
all_iou[:,0] = 0 # ignore the dummy label

# Now compute the highest-iou label for every unit
unit_label_99 = [
        (concept.item(), seglabels[concept], segcatlabels[concept], bestiou.item())
        for (bestiou, concept) in zip(*all_iou.max(1))]
label_list = [labelcat for concept, label, labelcat, iou in unit_label_99 if iou > 0.04]
experiment.save_conceptcat_graph(resfile('unit_concepts.svg'), conceptcatlist)

# Now let's pull out the higest-activating units for one image.
image_number = 10000

# Grab the image
im = dataset[image_number][0]
# Run the network to get the activations
r = run(im[None])
# Now get the maximum value of each channel
fmax = r.view(512, 14*14).max(1)[0]
# Convert it to a percentile (optional)
fmax_as_percentile = rq.normalize(fmax)
# Take the values that are the highest
top_by_q = fmax_as_percentile.sort(0, descending=True)[1][:6]

# Now save some files
os.makedirs(resfile('im_{image_number}_example'))
activations = r[0]
for i, u in eumerate(top_by_q):
    iv.masked_images(im, activations, u.item(), level=level[u]).save(
            resfile('im_{image_number}_example/{i}_a_image_unit_{u}.jpeg'))
    unit_images[u].save(
            resfile('im_{image_number}_example/{i}_b_rep_unit_{u}.jpeg'))


