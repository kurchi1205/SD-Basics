import torch
import numpy as np
from typing import List

def variant_loss(images, rgb_value: List):
    """
    This function computes the variant loss for the given image and the rgb value.
    :param images: The image for which the variant loss is to be computed.
    :param rgb_value: The rgb value of the image.
    :return: The variant loss for the given image and the rgb value.
    """
    
    sum_rgb = np.sum(rgb_value)
    r_wt = rgb_value[0] / sum_rgb
    g_wt = rgb_value[1] / sum_rgb
    b_wt = rgb_value[2] / sum_rgb

    r_loss = r_wt * torch.abs(images[:, 0] - 0.9).mean()
    g_loss = g_wt * torch.abs(images[:, 1] - 0.9).mean()
    b_loss = b_wt * torch.abs(images[:, 2] - 0.9).mean()

    variant_loss = r_loss + g_loss + b_loss
    return variant_loss

