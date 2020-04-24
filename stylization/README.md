# stylize-datasets
This repository contains code for stylizing arbitrary image datasets using [AdaIN](https://arxiv.org/abs/1703.06868). The code is a generalization of Robert Geirhos' [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) code, which is tailored to stylizing ImageNet. Everything in this repository is based on naoto0804's [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) implementation.

Given an image dataset, the script creates the specified number of stylized versions of every image while keeping the directory structure and naming scheme intact (usefull for existing data loaders or if directory names include class annotations).

Feel free to open an issue in case there is any question.

## Usage
- Dependencies:
    - python >= 3.6
    - Pillow
    - torch
    - torchvision
    - tqdm  
- Download the models:
    - either run run `bash models/download_models.sh` or download the models manually from [vgg](https://drive.google.com/file/d/108uza-dsmwvbW2zv-G73jtVcMU_2Nb7Y/view)/[decoder](https://drive.google.com/file/d/1w9r1NoYnn7tql1VYG3qDUzkbIks24RBQ/view) and move both files to the `models/` directory
    - Get style images: Download train.zip from [Kaggle's painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data)
- To stylize a dataset, run `python stylize.py`.

    Arguments:
    - `--content-dir <CONTENT>` the top-level directory of the content image dataset (mandatory)
    - `--style-dir <STLYE>` the top-level directory of the style images (mandatory)
    - `--output-dir <OUTPUT>` the directory where the stylized dataset will be stored (optional, default: `output/`)
    - `--num-styles <N>` number of stylizations to create for each content image (optional, default: `1`)
    - `--alpha <A>` Weight that controls the strength of stylization, should be between 0 and 1 (optional, default: `1`)
    - `--extensions <EX0> <EX1> ...` list of image extensions to scan style and content directory for (optional, default: `png, jpeg, jpg`). Note: this is case sensitive, `--extensions jpg` will not scan for files ending on `.JPG`. Image types must be compatible with PIL's `Image.open()` ([Documentation](https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html))
    - `--content-size <N>` Minimum size for content images, resulting in scaling of the shorter side of the content image to `N` (optional, default: `0`). Set this to 0 to keep the original image dimensions.
    - `--style-size <N>` Minimum size for style images, resulting in scaling of the shorter side of the style image to `N` (optional, default: `512`). Set this to 0 to keep the original image dimensions (for large style images, this will result in high (GPU) memory consumption).
    - `--crop` If set, content and style images will be cropped at the center to create square output images

Here is an example call:

```
 python3 stylize.py --content-dir '/home/username/stylize-datasets/images/' --style-dir '/home/username/stylize-datasets/train/' --num-styles 10 --content_size 0 --style_size 256
 ```
