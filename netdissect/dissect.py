import torch
import re
import copy
import numpy
from torch.utils.data.dataloader import default_collate
from netdissect import nethook, imgviz, tally, unravelconv, upsample, renormalize


def acts_image(model, dataset,
               layer=None, unit=None,
               thumbsize=None,
               cachedir=None,
               return_as='strip',  # or individual, or tensor
               k=100, r=4096, q=0.01,
               batch_size=10,
               sample_size=None,
               num_workers=30):
    assert return_as in ['strip', 'individual', 'tensor']
    topk, rq, run = acts_stats(model, dataset, layer=layer, unit=unit,
                               k=max(200, k), r=r, batch_size=batch_size, num_workers=num_workers,
                               sample_size=sample_size, cachedir=cachedir)
    result = window_images(dataset, topk, rq, run,
                           thumbsize=thumbsize, return_as=return_as, k=k, q=q,
                           cachedir=cachedir)
    if unit is not None and not hasattr(unit, '__len__'):
        result = result[0]
    return result


def grad_image(model, dataset,
               layer=None, unit=None,
               thumbsize=None,
               cachedir=None,
               return_as='strip',  # or individual, or tensor
               k=100, r=4096, q=0.01,
               batch_size=10,
               sample_size=None,
               num_workers=30):
    assert return_as in ['strip', 'individual', 'tensor']
    topk, botk, rq, run = grad_stats(model, dataset, layer=layer, unit=unit,
                                     k=max(200, k), r=r, batch_size=batch_size, num_workers=num_workers,
                                     sample_size=sample_size, cachedir=cachedir)
    result = window_images(dataset, topk, rq, run,
                           thumbsize=thumbsize, return_as=return_as, k=k, q=q,
                           cachedir=cachedir)
    if unit is not None and not hasattr(unit, '__len__'):
        result = result[0]
    return result


def update_image(model, dataset,
                 layer=None, unit=None,
                 thumbsize=None,
                 cachedir=None,
                 return_as='strip',  # or individual, or tensor
                 k=100, r=4096, q=0.01,
                 cinv=None,
                 batch_size=10,
                 sample_size=None,
                 num_workers=30):
    assert return_as in ['strip', 'individual', 'tensor']
    topk, botk, rq, run = update_stats(model, dataset, layer=layer, unit=unit,
                                       k=max(200, k), r=r, cinv=cinv,
                                       batch_size=batch_size, num_workers=num_workers,
                                       sample_size=sample_size, cachedir=cachedir)
    result = window_images(dataset, topk, rq, run,
                           thumbsize=thumbsize, return_as=return_as, k=k, q=q,
                           cachedir=cachedir)
    if unit is not None and not hasattr(unit, '__len__'):
        result = result[0]
    return result


def proj_image(model, dataset,
               layer=None, unit=None,
               thumbsize=None,
               cachedir=None,
               return_as='strip',  # or individual, or tensor
               k=100, r=4096, q=0.01,
               batch_size=10,
               sample_size=None,
               num_workers=30):
    assert return_as in ['strip', 'individual', 'tensor']
    topk, botk, rq, run = proj_stats(model, dataset, layer=layer, unit=unit,
                                     k=max(200, k), r=r, batch_size=batch_size, num_workers=num_workers,
                                     sample_size=sample_size, cachedir=cachedir)
    result = window_images(dataset, topk, rq, run,
                           thumbsize=thumbsize, return_as=return_as, k=k, q=q,
                           cachedir=cachedir)
    if unit is not None and not hasattr(unit, '__len__'):
        result = result[0]
    return result


def acts_stats(model, dataset,
               layer=None, unit=None,
               cachedir=None,
               k=100, r=4096,
               batch_size=10,
               sample_size=None,
               num_workers=30):
    assert not model.training
    if unit is not None:
        if not hasattr(unit, '__len__'):
            unit = [unit]
    assert unit is None or len(unit) > 0
    if layer is not None:
        module = nethook.get_module(model, layer)
    else:
        module = model
    device = next(model.parameters()).device
    pin_memory = (device.type != 'cpu')

    def run(x, *args):
        with nethook.Trace(module, stop=True) as ret, torch.no_grad():
            model(x.to(device))
        r = ret.output
        if unit is not None:
            r = r[:, unit]
        return r
    run.name = 'acts'

    def compute_samples(batch, *args):
        r = run(batch)
        flat_r = r.view(r.shape[0], r.shape[1], -1)
        top_r = flat_r.max(2)[0]
        all_r = r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
        return top_r, all_r
    topk, rq = tally.tally_topk_and_quantile(
        compute_samples, dataset, k=k, r=r,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
        sample_size=sample_size,
        cachefile=f'{cachedir}/acts_topk_rq.npz' if cachedir else None)
    return topk, rq, run


def grad_stats(model, dataset, layer,
               unit=None,
               cachedir=None,
               k=100, r=4096,
               batch_size=10,
               sample_size=None,
               num_workers=30,
               ):
    assert not model.training
    if unit is not None:
        if not hasattr(unit, '__len__'):
            unit = [unit]
    assert unit is None or len(unit) > 0
    # Make a copy so we can disable grad on parameters
    cloned_model = copy.deepcopy(model)
    nethook.set_requires_grad(False, cloned_model)
    if layer is not None:
        module = nethook.get_module(cloned_model, layer)
    else:
        module = cloned_model
    device = next(cloned_model.parameters()).device
    pin_memory = (device.type != 'cpu')

    def run(x, y, *args):
        with nethook.Trace(module, retain_grad=True) as ret, (
                torch.enable_grad()):
            out = cloned_model(x.to(device))
            r = ret.output
            loss = torch.nn.functional.cross_entropy(out, y.to(device))
            loss.backward()
            r = -r.grad
            if unit is not None:
                r = r[:, unit]
            return r
    run.name = 'grad'

    def compute_samples(x, y, *args):
        r = run(x, y)
        flat_r = r.view(r.shape[0], r.shape[1], -1)
        top_r = flat_r.max(2)[0]
        bot_r = flat_r.min(2)[0]
        all_r = r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
        return top_r, bot_r, all_r
    topk, botk, rq = tally.tally_extremek_and_quantile(
        compute_samples, dataset, k=k, r=r,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
        sample_size=sample_size,
        cachefile=f'{cachedir}/grad_exk_rq.npz' if cachedir else None)
    return topk, botk, rq, run


def weight_grad(model, dataset, layer,
                unit=None,
                cachedir=None,
                batch_size=10,
                sample_size=None,
                num_workers=30):
    # Make a copy so we can disable grad on parameters
    cloned_model = copy.deepcopy(model)
    nethook.set_requires_grad(False, cloned_model)
    module = nethook.get_module(cloned_model, layer)
    nethook.set_requires_grad(True, module)
    device = next(cloned_model.parameters()).device
    pin_memory = (device.type != 'cpu')

    def accumulate_grad(x, y, *args):
        with torch.enable_grad():
            out = cloned_model(x.to(device))
            loss = torch.nn.functional.cross_entropy(out, y.to(device))
            loss.backward()

    def weight_grad():
        return dict(wgrad=module.weight.grad)
    module.weight.grad = None
    wg = tally.tally_each(accumulate_grad, dataset, summarize=weight_grad,
                          batch_size=batch_size,
                          num_workers=num_workers, pin_memory=pin_memory,
                          sample_size=sample_size,
                          cachefile=f'{cachedir}/weight_grad.npz' if cachedir else None)['wgrad']
    return wg


def update_stats(model, dataset, layer,
                 unit=None,
                 cachedir=None,
                 k=100, r=4096,
                 batch_size=10,
                 cinv=None,
                 sample_size=None,
                 num_workers=30,
                 ):
    assert not model.training
    if unit is not None:
        if not hasattr(unit, '__len__'):
            unit = [unit]
    assert unit is None or len(unit) > 0
    # get weight grad (assumes layer has a weight param)
    wg = weight_grad(model, dataset, layer,
                     cachedir=cachedir,
                     batch_size=batch_size,
                     sample_size=sample_size,
                     num_workers=num_workers)
    if cinv is not None:
        wg = torch.mm(wg.view(-1,
                              cinv.shape[0]).cpu(),
                      cinv.cpu()).view(wg.shape)
    # copy the model so we can change its weights.
    cloned_model = copy.deepcopy(model)
    nethook.set_requires_grad(False, cloned_model)
    module = nethook.get_module(cloned_model, layer)
    device = next(cloned_model.parameters()).device
    pin_memory = (device.type != 'cpu')
    with torch.no_grad():
        module.weight[...] = -wg.to(device)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias[...] = 0

    def run(x, *args):
        with nethook.Trace(module, stop=True) as ret, torch.no_grad():
            cloned_model(x.to(device))
        r = ret.output
        if unit is not None:
            r = r[:, unit]
        return r
    run.name = 'update' if cinv is None else 'proj'

    def compute_samples(batch, *args):
        r = run(batch)
        flat_r = r.view(r.shape[0], r.shape[1], -1)
        top_r = flat_r.max(2)[0]
        bot_r = flat_r.min(2)[0]
        all_r = r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
        return top_r, bot_r, all_r
    topk, botk, rq = tally.tally_extremek_and_quantile(
        compute_samples, dataset, k=k, r=r,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
        sample_size=sample_size,
        cachefile=f'{cachedir}/{run.name}_exk_rq.npz' if cachedir else None)
    return topk, botk, rq, run


def proj_c2m(model, dataset, layer,
             cachedir=None,
             batch_size=10,
             sample_size=None,
             num_workers=30,
             ):
    assert not model.training
    device = next(model.parameters()).device
    pin_memory = (device.type != 'cpu')
    cloned_model = copy.deepcopy(model)
    module = nethook.get_module(cloned_model, layer)
    assert isinstance(module, torch.nn.Conv2d)
    nethook.set_requires_grad(False, cloned_model)
    unraveled = unravelconv.unravel_left_conv2d(module)
    unraveled.wconv.weight.requires_grad = True
    unraveled.wconv.weight.grad = None
    nethook.replace_module(cloned_model, layer, unraveled)
    tconv = unraveled.tconv

    def ex_run(x, *args):
        with nethook.Trace(tconv, stop=True) as unrav:
            cloned_model(x.to(device))
        return unrav.output

    def ex_sample(x, *args):
        r = ex_run(x, *args)
        return r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
    c2m = tally.tally_second_moment(
        ex_sample,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
        sample_size=sample_size,
        cachefile=f'{cachedir}/input_cov_moment.npz' if cachedir else None)
    return c2m, ex_run


def proj_stats(model, dataset, layer,
               unit=None,
               cachedir=None,
               k=100, r=4096,
               batch_size=10,
               sample_size=None,
               num_workers=30,
               ):
    c2m, ex_run = proj_c2m(model, dataset, layer,
                           batch_size=batch_size, sample_size=sample_size,
                           cachedir=cachedir)
    # old obsolete method - not stable.
    # Cinv = c2m.momentPSD().cholesky_inverse()
    moment = c2m.moment()
    # TODO: consider uncommenting the following, which uses
    # correlation for a better-conditioned inverse.
    # Change 2.0 to 3.0 to reduce amplifying near-zero feats.
    # rn = moment.diag().clamp(1e-30).pow(-1/2.0)
    # moment = moment * rn[None,:] * rn[:,None]
    # The following is standard regularization, to try.
    # moment.diagonal.add_(1e-3)
    Cinv = moment.pinverse()

    return update_stats(model, dataset, layer, unit=unit,
                        cinv=Cinv,
                        k=k, r=r, batch_size=batch_size, sample_size=sample_size,
                        cachedir=cachedir)


def window_images(dataset, topk, rq, run,
                  thumbsize=None,
                  return_as='strip',  # or individual, or tensor
                  k=None, q=0.01,
                  border_color=None,
                  vizname=None,
                  cachedir=None):
    assert return_as in ['strip', 'individual', 'tensor']
    input_sample = default_collate([dataset[0]])
    r_sample = run(*input_sample)
    x_size = tuple(input_sample[0].shape[2:])
    if thumbsize is None:
        thumbsize = x_size
    if not isinstance(thumbsize, (list, tuple)):
        thumbsize = (thumbsize, thumbsize)
    if topk is None:
        topk = tally.range_topk(r_sample.size(1), size=(k or 1))
    default_vizname = 'top' if topk.largest else 'bot'
    if border_color in ['red', 'green', 'yellow']:
        default_vizname += border_color
        border_color = dict(red=[255.0, 0.0, 0.0], green=[0.0, 255.0, 0.0],
                            yellow=[255.0, 255.0, 0.0])[border_color]
    if vizname is None:
        vizname = default_vizname
    iv = imgviz.ImageVisualizer(
        thumbsize, image_size=x_size, source=dataset,
        level=rq.quantiles((1.0 - q) if topk.largest else q))
    func = dict(
        strip=iv.masked_images_for_topk,
        individual=iv.individual_masked_images_for_topk,
        tensor=iv.masked_image_grid_for_topk)[return_as]
    acts_images = func(run, dataset, topk, k=k, largest=topk.largest,
                       border_color=border_color,
                       cachefile=f'{cachedir}/{vizname}{k or ""}images.npz' if cachedir else None)
    return acts_images


def label_stats(dataset_with_seg, num_seglabels,
                run, level, upfn=None,
                negate=False,
                cachedir=None,
                batch_size=10,
                sample_size=None,
                num_workers=30):
    # Create upfn
    data_sample = default_collate([dataset_with_seg[0]])
    input_sample = data_sample[:-2] + data_sample[-1:]
    seg_sample = data_sample[-2]
    r_sample = run(*input_sample)
    r_size = tuple(r_sample.shape[2:])
    seg_size = tuple(seg_sample.shape[2:])
    device = r_sample.device
    pin_memory = (device.type != 'cpu')
    if upfn is None:
        upfn = upsample.upsampler(seg_size, r_size)

    def compute_concept_pair(batch, seg, *args):
        seg = seg.to(device)
        acts = run(batch, *args)
        hacts = upfn(acts)
        iacts = (hacts < level if negate else hacts > level)  # indicator
        iseg = torch.zeros(seg.shape[0], num_seglabels,
                           seg.shape[2], seg.shape[3],
                           dtype=torch.bool, device=seg.device)
        iseg.scatter_(dim=1, index=seg, value=1)
        flat_segs = iseg.permute(0, 2, 3, 1).reshape(-1, iseg.shape[1])
        flat_acts = iacts.permute(0, 2, 3, 1).reshape(-1, iacts.shape[1])
        return flat_segs, flat_acts
    neg = 'neg' if negate else ''
    iu99 = tally.tally_all_intersection_and_union(
        compute_concept_pair,
        dataset_with_seg,
        sample_size=sample_size,
        num_workers=num_workers, pin_memory=pin_memory,
        cachefile=f'{cachedir}/{neg}{run.name}_iu.npz' if cachedir else None)
    return iu99

def topk_label_stats(dataset, run, level, topk, k=None,
                     upfn=None,
                     negate=False,
                     cachedir=None,
                     batch_size=10,
                     sample_size=None,
                     num_workers=30):
    # Create upfn
    data_sample = default_collate([dataset[0]])
    r_sample, seg_sample = run(*data_sample)
    r_size = tuple(r_sample.shape[2:])
    seg_size = tuple(seg_sample.shape[2:])
    device = r_sample.device
    num_units = r_sample.shape[1]
    assert level.shape[0] == num_units and len(level.shape) == 1
    level = level[None,:,None,None].to(device)
    pin_memory = (device.type != 'cpu')
    num_seglabels = run.num_seglabels
    if upfn is None:
        upfn = upsample.upsampler(seg_size, r_size)
    intersections = torch.zeros(num_units, num_seglabels).to(device)
    unions = torch.zeros(num_units, num_seglabels).to(device)
    def collate_unit_iou(units, *batch):
        acts, seg = [d.to(device) for d in run(*batch)]
        hacts = upfn(acts)
        iacts = (hacts > level)  # indicator
        iseg = torch.zeros(seg.shape[0], num_seglabels,
                           seg.shape[2], seg.shape[3],
                           dtype=torch.bool, device=seg.device)
        iseg.scatter_(dim=1, index=seg, value=1)
        for i in range(len(iacts)):
            ulist = units[i]
            for unit, _ in ulist:
                im_i = (iacts[i, unit][None] & iseg[i]).view(
                    num_seglabels, -1).float().sum(1)
                im_u = (iacts[i, unit][None] | iseg[i]).view(
                    num_seglabels, -1).float().sum(1)
                intersections[unit] += im_i
                unions[unit] += im_u
        return []
    tally.gather_topk(collate_unit_iou, dataset, topk, k=100)
    return intersections / (unions + 1e-20)

def make_rws_for_segmenter(run, dataset, segmenter, downsample=1):
    renorm = renormalize.renormalizer(dataset, target='zc')
    seglabels, _ = segmenter.get_label_and_category_names()
    num_seglabels = len(seglabels)
    def rws(*batch):
        batch = [d.cuda() for d in batch]
        images = batch[0]
        seg_batch = segmenter.segment_batch(images, downsample=downsample)
        r = run(*batch)
        return r, seg_batch
    rws.num_seglabels = num_seglabels
    return rws

def topk_label_stats_using_segmodel(
        dataset, segmodel, run, level, topk, downsample=1, **kwargs):
    rws = make_rws_for_segmenter(run, dataset, segmodel, downsample=downsample)
    return topk_label_stats(dataset, rws, level, topk, **kwargs)

def make_rws_for_dataset_with_seg(run, num_seglabels):
    def rws(*batch_with_seg):
        input_batch = batch_with_seg[:1] + batch_with_seg[2:]
        seg_batch = batch_with_seg[2]
        r = run(input_batch)
        return r, seg_batch
    rws.num_seglabels = num_seglabels
    return rws

def topk_label_stats_using_dataseat_with_seg(
        dataset_with_seg, num_seglabels, run, level, topk, **kwargs):
    rws = make_rws_for_dataset_with_seg(run, num_seglabels)
    return topk_label_stats(dataset, rws, level, topk, **kwargs)
