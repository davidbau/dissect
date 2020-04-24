import argparse, os, json, numpy, PIL.Image, torch, torchvision, collections
import math, shutil
from netdissect import pidfile, tally, nethook, parallelfolder
from netdissect import upsample, imgviz, imgsave, renormalize, bargraph
from netdissect import runningstats


class DissectVis:
    '''
    Code to read out the dissection in a set of directories.
    '''
    def __init__(self, outdir='results', model='vgg16',
            dataset='places', layers=None,
            seg='netpqc', quantile=0.01):
        labels = {}
        iou = {}
        images = {}
        rq = {}
        dirs = {}
        for k in layers:
            dirname = os.path.join(outdir,
                    f"{model}-{dataset}-{seg}-{k}-{int(1000 * quantile)}")
            dirs[k] = dirname
            with open(os.path.join(dirname, 'report.json')) as f:
                labels[k] = json.load(f)['units']
            rq[k] = runningstats.RunningQuantile(
                    state=numpy.load(os.path.join(dirname, 'rq.npz'),
                        allow_pickle=True))
            images[k] = [None] * rq[k].depth
        self.dirs = dirs
        self.labels = labels
        self.rqtable = rq
        self.images = images
        self.basedir = outdir
        
    def label(self, layer, unit):
        return self.labels[layer][unit]['label']
    def iou(self, layer, unit):
        return self.labels[layer][unit]['iou']
    def dir(self, layer):
        return self.dirs[layer]
    def rq(self, layer):
        return self.rqtable[layer]
    def image(self, layer, unit):
        result = self.images[layer][unit]
        # Lazy loading of images.
        if result is None:
            result = PIL.Image.open(os.path.join(
                self.dirs[layer],
                'image/unit%d.jpg' % unit))
            result.load()
            self.images[layer][unit] = result
        return result

    def save_bargraph(self, filename, layer, min_iou=0.04):
        svg = self.bargraph(layer, min_iou=min_iou, file_header=True)
        with open(filename, 'w') as f:
            f.write(svg)

    def img_bargraph(self, layer, min_iou=0.04):
        url = self.bargraph(layer, min_iou=min_iou, data_url=True)
        class H:
            def __init__(self, url):
                self.url = url
            def _repr_html_(self):
                return '<img src="%s">' % self.url
        return H(url)

    def bargraph(self, layer, min_iou=0.04, **kwargs):
        labelcat_list = []
        for rec in self.labels[layer]:
            if rec['iou'] and rec['iou'] >= min_iou:
                labelcat_list.append(tuple(rec['cat']))
        return self.bargraph_from_conceptcatlist(labelcat_list, **kwargs)

    def bargraph_from_conceptcatlist(self, conceptcatlist, **kwargs):
        count = collections.defaultdict(int)
        catcount = collections.defaultdict(int)
        for c in conceptcatlist:
            count[c] += 1
        for c in count.keys():
            catcount[c[1]] += 1
        cats = ['object', 'part', 'material', 'texture', 'color']
        catorder = dict((c, i) for i, c in enumerate(cats))
        sorted_labels = sorted(count.keys(),
            key=lambda x: (catorder[x[1]], -count[x]))
        sorted_labels
        return bargraph.make_svg_bargraph(
            [label for label, cat in sorted_labels],
            [count[k] for k in sorted_labels],
            [(c, catcount[c]) for c in cats], **kwargs)

