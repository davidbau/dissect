# Measuring the importance of a unit to a class by measuring the
# impact of removing sets of units on binary classification
# accuracy for individual classes.

import torch, argparse, os, json, numpy, random
from netdissect import pbar, nethook
from netdissect.sampler import FixedSubsetSampler
from . import setting
import netdissect
torch.backends.cudnn.benchmark = True

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', choices=['vgg16'], default='vgg16')
    aa('--dataset', choices=['places'], default='places')
    aa('--layer', default='features.conv5_3')
    args = parser.parse_args()
    return args

def main():
    args = parseargs()

    model = setting.load_classifier(args.model)
    model = nethook.InstrumentedModel(model).cuda().eval()
    layername = args.layer
    model.retain_layer(layername)
    dataset = setting.load_dataset(args.dataset, crop_size=224)
    train_dataset = setting.load_dataset(args.dataset, crop_size=224,
            split='train')
    sample_size = len(dataset)

    # Probe layer to get sizes
    model(dataset[0][0][None].cuda())
    num_units = model.retained_layer(layername).shape[1]
    classlabels = dataset.classes

    # Measure baseline classification accuracy on val set, and cache.
    pbar.descnext('baseline_pra')
    baseline_precision, baseline_recall, baseline_accuracy, baseline_ba  = (
        test_perclass_pra(
            model, dataset,
            cachefile=sharedfile('pra-%s-%s/pra_baseline.npz'
                % (args.model, args.dataset))))
    pbar.print('baseline acc', baseline_ba.mean().item())

    # Now erase each unit, one at a time, and retest accuracy.
    unit_list = random.sample(list(range(num_units)), num_units)
    val_single_unit_ablation_ba = torch.zeros(num_units, len(classlabels))
    for unit in pbar(unit_list):
        pbar.descnext('test unit %d' % unit)
        # Get binary accuracy if the model after ablating the unit.
        _, _, _, ablation_ba = test_perclass_pra(
                model, dataset,
                layername=layername,
                ablated_units=[unit],
                cachefile=sharedfile('pra-%s-%s/pra_ablate_unit_%d.npz' %
                    (args.model, args.dataset, unit)))
        val_single_unit_ablation_ba[unit] = ablation_ba

    # For the purpose of ranking units by importance to a class, we
    # measure using the training set (to avoid training unit ordering
    # on the test set).
    sample_size = None
    # Measure baseline classification accuracy, and cache.
    pbar.descnext('train_baseline_pra')
    baseline_precision, baseline_recall, baseline_accuracy, baseline_ba  = (
        test_perclass_pra(
            model, train_dataset,
            sample_size=sample_size,
            cachefile=sharedfile('ttv-pra-%s-%s/pra_train_baseline.npz'
                % (args.model, args.dataset))))
    pbar.print('baseline acc', baseline_ba.mean().item())

    # Measure accuracy on the val set.
    pbar.descnext('val_baseline_pra')
    _, _, _, val_baseline_ba  = (
        test_perclass_pra(
            model, dataset,
            cachefile=sharedfile('ttv-pra-%s-%s/pra_val_baseline.npz'
                % (args.model, args.dataset))))
    pbar.print('val baseline acc', val_baseline_ba.mean().item())

    # Do in shuffled order to allow multiprocessing.
    single_unit_ablation_ba = torch.zeros(num_units, len(classlabels))
    for unit in pbar(unit_list):
        pbar.descnext('test unit %d' % unit)
        _, _, _, ablation_ba = test_perclass_pra(
                model, train_dataset,
                layername=layername,
                ablated_units=[unit],
                sample_size=sample_size,
                cachefile=
                    sharedfile('ttv-pra-%s-%s/pra_train_ablate_unit_%d.npz' %
                    (args.model, args.dataset, unit)))
        single_unit_ablation_ba[unit] = ablation_ba

    # Now for every class, remove a set of the N most-important
    # and N least-important units for that class, and measure accuracy.
    for classnum in pbar(random.sample(range(len(classlabels)),
            len(classlabels))):
        # For a few classes, let's chart the whole range of ablations.
        if classnum in [100, 169, 351, 304]:
            num_best_list = range(1, num_units)
        else:
            num_best_list = [1, 2, 3, 4, 5, 20, 64, 128, 256]
        pbar.descnext('numbest')
        for num_best in pbar(random.sample(num_best_list, len(num_best_list))):
            num_worst = num_units - num_best
            unitlist = single_unit_ablation_ba[:,classnum].sort(0)[1][:num_best]
            _, _, _, testba = test_perclass_pra(model, dataset,
                layername=layername,
                ablated_units=unitlist,
                cachefile=sharedfile(
                    'ttv-pra-%s-%s/pra_val_ablate_classunits_%s_ba_%d.npz'
                    % (args.model, args.dataset, classlabels[classnum],
                        len(unitlist))))
            unitlist = (
                single_unit_ablation_ba[:,classnum].sort(0)[1][-num_worst:])
            _, _, _, testba2 = test_perclass_pra(model, dataset,
                layername=layername,
                ablated_units=unitlist,
                cachefile=sharedfile(
                  'ttv-pra-%s-%s/pra_val_ablate_classunits_%s_worstba_%d.npz' %
                    (args.model, args.dataset, classlabels[classnum],
                        len(unitlist))))
            pbar.print('%s: best %d %.3f vs worst N %.3f' %
                    (classlabels[classnum], num_best,
                        testba[classnum] - val_baseline_ba[classnum],
                        testba2[classnum] - val_baseline_ba[classnum]))

def test_perclass_pra(model, dataset,
        layername=None, ablated_units=None, sample_size=None, cachefile=None):
    '''Classifier precision/recall/accuracy measurement.
    Disables a set of units in the specified layer, and then
    measures per-class precision, recall, accuracy and
    balanced (binary classification) accuracy for each class,
    compared to the ground truth in the given dataset.'''
    try:
        if cachefile is not None:
            data = numpy.load(cachefile)
            # verify that this is computed.
            data['true_negative_rate']
            result = tuple(torch.tensor(data[key]) for key in
                 ['precision', 'recall', 'accuracy', 'balanced_accuracy'])
            pbar.print('Loading cached %s' % cachefile)
            return result
    except:
        pass
    model.remove_edits()
    if ablated_units is not None:
        def ablate_the_units(x, *args):
            x[:,ablated_units] = 0
            return x
        model.edit_layer(layername, rule=ablate_the_units)
    with torch.no_grad():
        num_classes = len(dataset.classes)
        true_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
        pred_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
        correct_counts = torch.zeros(num_classes, dtype=torch.int64).cuda()
        total_count = 0
        sampler = None if sample_size is None else (
            FixedSubsetSampler(list(range(sample_size))))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=100, num_workers=20,
            sampler=sampler, pin_memory=True)
        for image_batch, class_batch in pbar(loader):
            total_count += len(image_batch)
            image_batch, class_batch = [
                d.cuda() for d in [image_batch, class_batch]]
            scores = model(image_batch)
            preds = scores.max(1)[1]
            correct = (preds == class_batch)
            true_counts.add_(class_batch.bincount(minlength=num_classes))
            pred_counts.add_(preds.bincount(minlength=num_classes))
            correct_counts.add_(class_batch.bincount(
                correct, minlength=num_classes).long())
    model.remove_edits()
    true_neg_counts = (
            (total_count - true_counts) - (pred_counts - correct_counts))
    precision = (correct_counts.float() / pred_counts.float()).cpu()
    recall = (correct_counts.float() / true_counts.float()).cpu()
    accuracy = (correct_counts + true_neg_counts).float().cpu() / total_count
    true_neg_rate = (true_neg_counts.float() /
            (total_count - true_counts).float()).cpu()
    balanced_accuracy = (recall + true_neg_rate) / 2
    if cachefile is not None:
        numpy.savez(cachefile,
                precision=precision.numpy(),
                recall=recall.numpy(),
                accuracy=accuracy.numpy(),
                true_negative_rate=true_neg_rate.numpy(),
                balanced_accuracy=balanced_accuracy.numpy())
    return precision, recall, accuracy, balanced_accuracy

def sharedfile(fn):
    filename = os.path.join('results/shared', fn)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

if __name__ == '__main__':
    main()

