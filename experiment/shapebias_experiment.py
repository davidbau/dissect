from netdissect import parallelfolder, show, tally, nethook, renormalize
from . import readdissect, setting
import copy, PIL.Image
from netdissect import upsample, imgsave, imgviz
import re, torchvision, torch, os
from IPython.display import SVG
from matplotlib import pyplot as plt

torch.set_grad_enabled(False)

def normalize_filename(n):
    return re.match(r'^(.*Places365_\w+_\d+)', n).group(1)

ds = parallelfolder.ParallelImageFolders(
    ['datasets/places/val', 'datasets/stylized-places/val'],
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # transforms.CenterCrop(224),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        renormalize.NORMALIZER['imagenet'],
    ]),
    normalize_filename=normalize_filename,
    shuffle=True)


layers = [
    'conv5_3',
    'conv5_2',
    'conv5_1',
    'conv4_3',
    'conv4_2',
    'conv4_1',
    'conv3_3',
    'conv3_2',
    'conv3_1',
    'conv2_2',
    'conv2_1',
    'conv1_2',
    'conv1_1',
]
qd = readdissect.DissectVis(layers=layers)
net = setting.load_classifier('vgg16')

sds = parallelfolder.ParallelImageFolders(
    ['datasets/stylized-places/val'],
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # transforms.CenterCrop(224),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        renormalize.NORMALIZER['imagenet'],
    ]),
    normalize_filename=normalize_filename,
    shuffle=True)

uds = parallelfolder.ParallelImageFolders(
    ['datasets/places/val'],
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # transforms.CenterCrop(224),
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        renormalize.NORMALIZER['imagenet'],
    ]),
    normalize_filename=normalize_filename,
    shuffle=True)

# def s_image(layername, unit):
#    result = PIL.Image.open(os.path.join(qd.dir(layername), 's_imgs/unit_%d.png' % unit))
#    result.load()
#    return result

for layername in layers:
    #if os.path.isfile(os.path.join(qd.dir(layername), 'intersect_99.npz')):
    #    continue
    busy_fn = os.path.join(qd.dir(layername), 'busy.txt')
    if os.path.isfile(busy_fn):
        print(busy_fn)
        continue
    with open(busy_fn, 'w') as f:
        f.write('busy')
    print('working on', layername)

    inst_net = nethook.InstrumentedModel(copy.deepcopy(net)).cuda()
    inst_net.retain_layer('features.' + layername)
    inst_net(ds[0][0][None].cuda())
    sample_act = inst_net.retained_layer('features.' + layername).cpu()
    upfn = upsample.upsampler((64, 64), sample_act.shape[2:])

    def flat_acts(batch):
        inst_net(batch.cuda())
        acts = upfn(inst_net.retained_layer('features.' + layername))
        return acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    s_rq = tally.tally_quantile(flat_acts, sds, cachefile=os.path.join(
        qd.dir(layername), 's_rq.npz'))
    u_rq = qd.rq(layername)

    def intersect_99_fn(uimg, simg):
        s_99 = s_rq.quantiles(0.99)[None,:,None,None].cuda()
        u_99 = u_rq.quantiles(0.99)[None,:,None,None].cuda()
        with torch.no_grad():
            ux, sx = uimg.cuda(), simg.cuda()
            inst_net(ux)
            ur = inst_net.retained_layer('features.' + layername)
            inst_net(sx)
            sr = inst_net.retained_layer('features.' + layername)
            return ((sr > s_99).float() * (ur > u_99).float()
                    ).permute(0, 2, 3, 1).reshape(-1, ur.size(1))
    
    intersect_99 = tally.tally_mean(intersect_99_fn, ds,
        cachefile=os.path.join(qd.dir(layername), 'intersect_99.npz'))

    def compute_image_max(batch):
        inst_net(batch.cuda())
        return inst_net.retained_layer(
                'features.' + layername).max(3)[0].max(2)[0]

    s_topk = tally.tally_topk(compute_image_max, sds,
            cachefile=os.path.join(qd.dir(layername), 's_topk.npz'))

    def compute_acts(image_batch):
        inst_net(image_batch.cuda())
        acts_batch = inst_net.retained_layer('features.' + layername)
        return (acts_batch, image_batch)
    
    iv = imgviz.ImageVisualizer(128, quantiles=s_rq, source=sds)
    unit_images = iv.masked_images_for_topk(compute_acts, sds, s_topk, k=5)
    os.makedirs(os.path.join(qd.dir(layername),'s_imgs'), exist_ok=True)
    imgsave.save_image_set(unit_images,
         os.path.join(qd.dir(layername),'s_imgs/unit%d.jpg'))

    iv = imgviz.ImageVisualizer(128, quantiles=u_rq, source=uds)
    unit_images = iv.masked_images_for_topk(compute_acts, uds, s_topk, k=5)
    os.makedirs(os.path.join(qd.dir(layername),'su_imgs'), exist_ok=True)
    imgsave.save_image_set(unit_images,
         os.path.join(qd.dir(layername),'su_imgs/unit%d.jpg'))

    os.remove(busy_fn)
