# New-style dissection experiment code.
import torch, argparse, os, shutil, inspect, json, numpy, random
from collections import defaultdict
from netdissect import pbar, nethook, renormalize, zdataset
from netdissect import upsample, tally, imgviz, imgsave, bargraph
from . import setting
import netdissect
torch.backends.cudnn.benchmark = True

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', default='progan')
    aa('--dataset', default='church')
    aa('--seg', default='netpqc')
    aa('--layer', default='layer4')
    aa('--quantile', type=float, default=0.01)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    resdir = 'results/%s-%s-%s-%s-%s' % (
            args.model, args.dataset, args.seg, args.layer,
            int(args.quantile * 1000))
    def resfile(f):
        return os.path.join(resdir, f)

    model = load_model(args)
    layername = instrumented_layername(args)
    model.retain_layer(layername)
    dataset = load_dataset(args, model=model.model)
    upfn = make_upfn(args, dataset, model, layername)
    sample_size = len(dataset)
    is_generator = (args.model == 'progan')
    percent_level = 1.0 - args.quantile

    # Tally rq.np (representation quantile, unconditional).
    torch.set_grad_enabled(False)
    pbar.descnext('rq')
    def compute_samples(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    rq = tally.tally_quantile(compute_samples, dataset,
                              sample_size=sample_size,
                              r=8192,
                              num_workers=100,
                              pin_memory=True,
                              cachefile=resfile('rq.npz'))

    # Grab the 99th percentile, and tally conditional means at that level.
    level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]

    segmodel, seglabels, segcatlabels = setting.load_segmenter(args.seg)
    renorm = renormalize.renormalizer(dataset, target='zc')
    def compute_conditional_indicator(batch, *args):
        data_batch = batch.cuda()
        out_batch = model(data_batch)
        image_batch = out_batch if is_generator else renorm(data_batch)
        seg = segmodel.segment_batch(image_batch, downsample=4)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        iacts = (hacts > level_at_99).float() # indicator
        return tally.conditional_samples(iacts, seg)

    pbar.descnext('condi99')
    condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,
            num_workers=3, pin_memory=True,
            cachefile=resfile('condi99.npz'))

    # Now summarize the iou stats and graph the units
    iou_99 = tally.iou_from_conditional_indicator_mean(condi99)
    unit_label_99 = [
            (concept.item(), seglabels[concept],
                segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*iou_99.max(0))]

    def measure_segclasses_with_zeroed_units(zeroed_units, sample_size=100):
        model.remove_edits()
        def zero_some_units(x, *args):
            x[:, zeroed_units] = 0
            return x
        model.edit_layer(layername, rule=zero_some_units)
        num_seglabels = len(segmodel.get_label_and_category_names()[0])
        def compute_mean_seg_in_images(batch_z, *args):
            img = model(batch_z.cuda())
            seg = segmodel.segment_batch(img, downsample=4)
            seg_area = seg.shape[2] * seg.shape[3]
            seg_counts = torch.bincount((seg + (num_seglabels *
                torch.arange(seg.shape[0],
                    dtype=seg.dtype, device=seg.device
                    )[:,None,None,None])).view(-1),
                minlength=num_seglabels * seg.shape[0]).view(seg.shape[0], -1)
            seg_fracs = seg_counts.float() / seg_area
            return seg_fracs
        result = tally.tally_mean(compute_mean_seg_in_images, dataset,
                                batch_size=30, sample_size=sample_size, pin_memory=True)
        model.remove_edits()
        return result

    # Intervention experiment here:
    # segs_baseline = measure_segclasses_with_zeroed_units([])
    # segs_without_treeunits = measure_segclasses_with_zeroed_units(tree_units)
    num_units = len(unit_label_99)
    baseline_segmean = test_generator_segclass_stats(
                model, dataset, segmodel,
                layername=layername,
                cachefile=resfile('segstats/baseline.npz')).mean()

    pbar.descnext('unit ablation')
    unit_ablation_segmean = torch.zeros(num_units, len(baseline_segmean))
    for unit in pbar(random.sample(range(num_units), num_units)):
        stats = test_generator_segclass_stats(model, dataset, segmodel,
            layername=layername, zeroed_units=[unit],
            cachefile=resfile('segstats/ablated_unit_%d.npz' % unit))
        unit_ablation_segmean[unit] = stats.mean()

    ablate_segclass_name = 'tree'
    ablate_segclass = seglabels.index(ablate_segclass_name)
    best_iou_units = iou_99[ablate_segclass,:].sort(0)[1].flip(0)
    byiou_unit_ablation_seg = torch.zeros(30)
    for unitcount in pbar(random.sample(range(0,30), 30)):
        zero_units = best_iou_units[:unitcount].tolist()
        stats = test_generator_segclass_delta_stats(
            model, dataset, segmodel,
            layername=layername, zeroed_units=zero_units,
            cachefile=resfile('deltasegstats/ablated_best_%d_iou_%s.npz' %
                        (unitcount, ablate_segclass_name)))
        byiou_unit_ablation_seg[unitcount] = stats.mean()[ablate_segclass]

    # Generator context experiment.
    num_segclass = len(seglabels)
    door_segclass = seglabels.index('door')
    door_units = iou_99[door_segclass].sort(0)[1].flip(0)[:20]
    door_high_values = rq.quantiles(0.995)[door_units].cuda()

    def compute_seg_impact(zbatch, *args):
        zbatch = zbatch.cuda()
        model.remove_edits()
        orig_img = model(zbatch)
        orig_seg = segmodel.segment_batch(orig_img, downsample=4)
        orig_segcount = tally.batch_bincount(orig_seg, num_segclass)
        rep = model.retained_layer(layername).clone()
        ysize = orig_seg.shape[2] // rep.shape[2]
        xsize = orig_seg.shape[3] // rep.shape[3]
        def gen_conditions():
            for y in range(rep.shape[2]):
                for x in range(rep.shape[3]):
                    # Take as the context location the segmentation
                    # labels at the center of the square.
                    selsegs = orig_seg[:,:,y*ysize+ysize//2,
                            x*xsize+xsize//2]
                    changed_rep = rep.clone()
                    changed_rep[:,door_units,y,x] = (
                            door_high_values[None,:])
                    model.edit_layer(layername,
                            ablation=1.0, replacement=changed_rep)
                    changed_img = model(zbatch)
                    changed_seg = segmodel.segment_batch(
                            changed_img, downsample=4)
                    changed_segcount = tally.batch_bincount(
                            changed_seg, num_segclass)
                    delta_segcount = (changed_segcount
                            - orig_segcount).float()
                    for sel, delta in zip(selsegs, delta_segcount):
                        for cond in torch.bincount(sel).nonzero()[:,0]:
                            if cond == 0:
                                continue
                            yield (cond.item(), delta)
        return gen_conditions()

    cond_changes = tally.tally_conditional_mean(
            compute_seg_impact, dataset, sample_size=10000, batch_size=20,
            cachefile=resfile('big_door_cond_changes.npz'))

def test_generator_segclass_stats(model, dataset, segmodel,
        layername=None, zeroed_units=None, sample_size=None, cachefile=None):
    model.remove_edits()
    if zeroed_units is not None:
        def zero_some_units(x, *args):
            x[:, zeroed_units] = 0
            return x
        model.edit_layer(layername, rule=zero_some_units)
    num_seglabels = len(segmodel.get_label_and_category_names()[0])
    def compute_mean_seg_in_images(batch_z, *args):
        img = model(batch_z.cuda())
        seg = segmodel.segment_batch(img, downsample=4)
        seg_area = seg.shape[2] * seg.shape[3]
        seg_counts = torch.bincount((seg + (num_seglabels *
            torch.arange(seg.shape[0], dtype=seg.dtype, device=seg.device
                )[:,None,None,None])).view(-1),
            minlength=num_seglabels * seg.shape[0]).view(seg.shape[0], -1)
        seg_fracs = seg_counts.float() / seg_area
        return seg_fracs
    result = tally.tally_mean(compute_mean_seg_in_images, dataset,
                            batch_size=25, sample_size=sample_size,
                            pin_memory=True, cachefile=cachefile)
    model.remove_edits()
    return result

def make_upfn(args, dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None
    if args.model == 'alexnet':
        convs = [layer for name, layer in model.model.named_children()
                if name.startswith('conv') or name.startswith('pool')]
    elif args.model == 'progan':
        # Probe the data shape
        out = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        upfn = upsample.upsampler(
                (64, 64),
                data_shape=data_shape,
                image_size=out.shape[2:])
        return upfn
    else:
        # Probe the data shape
        _ = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        pbar.print('upsampling from data_shape', tuple(data_shape))
    upfn = upsample.upsampler(
            (56, 56),
            data_shape=data_shape,
            source=dataset,
            convolutions=convs)
    return upfn

def instrumented_layername(args):
    '''Chooses the layer name to dissect.'''
    if args.layer is not None:
        if args.model == 'vgg16':
            return 'features.' + args.layer
        return args.layer
    # Default layers to probe
    if args.model == 'alexnet':
        return 'conv5'
    elif args.model == 'vgg16':
        return 'features.conv5_3'
    elif args.model == 'resnet152':
        return '7'
    elif args.model == 'progan':
        return 'layer4'

def load_model(args):
    '''Loads one of the benchmark classifiers or generators.'''
    if args.model in ['alexnet', 'vgg16', 'resnet152']:
        model = setting.load_classifier(args.model)
    elif args.model == 'progan':
        model = setting.load_proggan(args.dataset)
    model = nethook.InstrumentedModel(model).cuda().eval()
    return model

def load_dataset(args, model=None):
    '''Loads an input dataset for testing.'''
    from torchvision import transforms
    if args.model == 'progan':
        dataset = zdataset.z_dataset_for_model(model, size=10000, seed=1)
        return dataset
    elif args.dataset in ['places']:
        crop_size = 227 if args.model == 'alexnet' else 224
        return setting.load_dataset(args.dataset, split='val', full=True,
                crop_size=crop_size, download=True)
    assert False

def graph_conceptcatlist(conceptcatlist, **kwargs):
    count = defaultdict(int)
    catcount = defaultdict(int)
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

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)

def test_generator_segclass_stats(model, dataset, segmodel,
        layername=None, zeroed_units=None, sample_size=None, cachefile=None):
    model.remove_edits()
    if zeroed_units is not None:
        def zero_some_units(x, *args):
            x[:, zeroed_units] = 0
            return x
        model.edit_layer(layername, rule=zero_some_units)
    num_seglabels = len(segmodel.get_label_and_category_names()[0])
    def compute_mean_seg_in_images(batch_z, *args):
        img = model(batch_z.cuda())
        seg = segmodel.segment_batch(img, downsample=4)
        seg_area = seg.shape[2] * seg.shape[3]
        seg_counts = torch.bincount((seg + (num_seglabels *
            torch.arange(seg.shape[0], dtype=seg.dtype, device=seg.device
                )[:,None,None,None])).view(-1),
            minlength=num_seglabels * seg.shape[0]).view(seg.shape[0], -1)
        seg_fracs = seg_counts.float() / seg_area
        return seg_fracs
    result = tally.tally_mean(compute_mean_seg_in_images, dataset,
                            batch_size=25, sample_size=sample_size,
                            pin_memory=True, cachefile=cachefile)
    model.remove_edits()
    return result


def test_generator_segclass_delta_stats(model, dataset, segmodel,
        layername=None, zeroed_units=None, sample_size=None, cachefile=None):
    model.remove_edits()
    def zero_some_units(x, *args):
        x[:, zeroed_units] = 0
        return x
    num_seglabels = len(segmodel.get_label_and_category_names()[0])
    def compute_mean_delta_seg_in_images(batch_z, *args):
        # First baseline
        model.remove_edits()
        img = model(batch_z.cuda())
        seg = segmodel.segment_batch(img, downsample=4)
        seg_area = seg.shape[2] * seg.shape[3]
        seg_counts = torch.bincount((seg + (num_seglabels *
            torch.arange(seg.shape[0], dtype=seg.dtype, device=seg.device
                )[:,None,None,None])).view(-1),
            minlength=num_seglabels * seg.shape[0]).view(seg.shape[0], -1)
        seg_fracs = seg_counts.float() / seg_area
        # Then with changes
        model.edit_layer(layername, rule=zero_some_units)
        d_img = model(batch_z.cuda())
        d_seg = segmodel.segment_batch(d_img, downsample=4)
        d_seg_counts = torch.bincount((d_seg + (num_seglabels *
            torch.arange(seg.shape[0], dtype=seg.dtype, device=seg.device
                )[:,None,None,None])).view(-1),
            minlength=num_seglabels * seg.shape[0]).view(seg.shape[0], -1)
        d_seg_fracs = d_seg_counts.float() / seg_area
        return d_seg_fracs - seg_fracs
    result = tally.tally_mean(compute_mean_delta_seg_in_images, dataset,
                            batch_size=25, sample_size=sample_size,
                            pin_memory=True, cachefile=cachefile)
    model.remove_edits()
    return result
class FloatEncoder(json.JSONEncoder):
    def __init__(self, nan_str='"NaN"', **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        def floatstr(o, allow_nan=self.allow_nan,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY,
                nan_str=self.nan_str):
            if o != o:
                text = nan_str
            elif o == _inf:
                text = '"Infinity"'
            elif o == _neginf:
                text = '"-Infinity"'
            else:
                return repr(o)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))
            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

def dump_json_file(target, data):
    with open(target, 'w') as f:
        json.dump(data, f, indent=1, cls=FloatEncoder)

def copy_static_file(source, target):
    sourcefile = os.path.join(
            os.path.dirname(inspect.getfile(netdissect)), source)
    shutil.copy(sourcefile, target)

if __name__ == '__main__':
    main()

