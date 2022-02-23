import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid

def tensor2im(image_tensor, imtype=np.float32, min_max=(-1, 1)):
    image_numpy = image_tensor.squeeze().cpu().float().numpy()
    image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = len(image_numpy.shape)
    if n_dim == 4:
        nc, nd, nh, nw = image_numpy.shape
        image_numpy = np.transpose(image_numpy[:, int(nd / 2)], (1, 2, 0))
        image_numpy -= np.amin(image_numpy)
        image_numpy /= np.amax(image_numpy)
    elif n_dim == 3:
        nd, nh, nw = image_numpy.shape
        image_numpy = image_numpy[int(nd / 2)].reshape(nh, nw, 1)
        image_numpy = np.tile(image_numpy, (1, 1, 3))
    elif n_dim == 2:
        nh, nw = image_numpy.shape
        image_numpy = image_numpy.reshape(nh, nw, 1)
        image_numpy = np.tile(image_numpy, (1, 1, 3))

    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
