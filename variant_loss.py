import torch
from typing import List

def variant_loss(images, rgb_value: List):
    """
    This function computes the variant loss for the given image and the rgb value.
    :param images: The image for which the variant loss is to be computed.
    :param rgb_value: The rgb value of the image.
    :return: The variant loss for the given image and the rgb value.
    """
    
    r_wt = rgb_value[0] / 255
    g_wt = rgb_value[1] / 255
    b_wt = rgb_value[2] / 255

    r_loss = 1 * torch.abs(images[:, 0] - r_wt).mean()
    g_loss = 1 * torch.abs(images[:, 1] - g_wt).mean()
    b_loss = 1 * torch.abs(images[:, 2] - b_wt).mean()

    variant_loss = r_loss + g_loss + b_loss
    return variant_loss

