import torch
from collections import OrderedDict


class RavelRightConv2d(torch.nn.Conv2d):
    '''
    A fixed "universal convolution" made of all ones and zeros, whiich
    reduces a c*ks-channel featuremap down to a c-channel featuremap by
    convolving channels together, with each channel shifted by a one-hot
    convolutional kernel pattern that corresponds to the standard
    arrangement of weights in a Conv2d.  Can optionally add a
    parameterized bias.
    '''

    def __init__(self, out_channels, kernel_size, padding, bias):
        kh, kw = kernel_size
        ks = kh * kw
        in_channels = out_channels * ks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias)
        weight = torch.zeros_like(self.weight)
        # Define the fixed convolution as a constant buffer.
        for y in range(kh):
            for x in range(kw):
                weight[:, x + kw * y:in_channels:ks,
                       y, x] = torch.eye(out_channels)
        del self.weight
        try:
            self.register_buffer(
                'weight', weight, persistent=False)  # pytorch >= 1.6
        except BaseException:
            self.register_buffer('weight', weight)


class RavelLeftConv2d(torch.nn.Conv2d):
    '''
    Just like RavelRight, but for use before a 1x1 operation instead of after.
    '''

    def __init__(self, in_channels, kernel_size, padding):
        kh, kw = kernel_size
        ks = kh * kw
        out_channels = in_channels * ks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)
        # Define the fixed convolution as a constant buffer.
        weight = torch.zeros_like(self.weight)
        for y in range(kh):
            for x in range(kw):
                weight[x + kw * y:out_channels:ks,
                       :, y, x] = torch.eye(in_channels)
        del self.weight
        try:
            self.register_buffer(
                'weight', weight, persistent=False)  # pytorch >= 1.6
        except BaseException:
            self.register_buffer('weight', weight)


def unravel_right_conv2d(original_conv):
    '''
    Converts a Conv2d to a sequence where a 1x1 convolution (dense per-pixel
    linear operation) "wconv" is followed by a fixed convolution "sconv".
    '''
    assert isinstance(original_conv, torch.nn.Conv2d)
    kh, kw = original_conv.kernel_size
    ks = kh * kw
    has_bias = hasattr(
        original_conv,
        'bias') and original_conv.bias is not None
    unraveled_conv = torch.nn.Sequential(OrderedDict([
        ('wconv', torch.nn.Conv2d(
            original_conv.in_channels,
            original_conv.out_channels * ks,
            kernel_size=1,
            stride=original_conv.stride,
            bias=False)),
        ('sconv', RavelRightConv2d(
            original_conv.out_channels,
            (kh, kw),
            original_conv.padding,
            has_bias))
    ]))
    with torch.no_grad():
        unraveled_conv.to(original_conv.weight)
        unraveled_conv.wconv.weight[...] = (original_conv.weight.permute(0, 2, 3, 1).reshape(
            original_conv.out_channels * ks, original_conv.in_channels, 1, 1))
        unraveled_conv.wconv.weight.requires_grad = original_conv.weight.requires_grad
        if has_bias:
            unraveled_conv.sconv.bias[...] = original_conv.bias
            unraveled_conv.sconv.bias.requires_grad = original_conv.bias.requires_grad
        if not original_conv.training:
            unraveled_conv.eval()
    return unraveled_conv


def unravel_left_conv2d(original_conv):
    '''
    Converts a Conv2d to a sequence where a fixed convolution "tconv" is followed
    by a dense 1x1 convolution (dense per-pixel linear operation) "wconv".
    '''
    assert isinstance(original_conv, torch.nn.Conv2d)
    kh, kw = original_conv.kernel_size
    ks = kh * kw
    has_bias = hasattr(
        original_conv,
        'bias') and original_conv.bias is not None
    unraveled_conv = torch.nn.Sequential(OrderedDict([
        ('tconv', RavelLeftConv2d(
            original_conv.in_channels,
            (kh, kw),
            original_conv.padding)),
        ('wconv', torch.nn.Conv2d(
            original_conv.in_channels * ks,
            original_conv.out_channels,
            kernel_size=1,
            stride=original_conv.stride,
            bias=has_bias))
    ]))
    with torch.no_grad():
        unraveled_conv.to(original_conv.weight)
        unraveled_conv.wconv.weight[...] = (original_conv.weight.reshape(
            original_conv.out_channels, original_conv.in_channels * ks, 1, 1))
        unraveled_conv.wconv.weight.requires_grad = original_conv.weight.requires_grad
        if has_bias:
            unraveled_conv.wconv.bias[...] = original_conv.bias
            unraveled_conv.wconv.bias.requires_grad = original_conv.bias.requires_grad
        if not original_conv.training:
            unraveled_conv.eval()
    return unraveled_conv
