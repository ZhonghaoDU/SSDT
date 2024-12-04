from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from utils.losses import DiceLoss

tDICE = DiceLoss(n_classes=3)
DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')
mse = nn.MSELoss()


def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(
        img_z * mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    loss_mask[:, w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 0
    return mask.long(), loss_mask.long()


def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * 2 / 3), int(img_y * 2 / 3), int(img_z * 2 / 3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x / 3) + 1, int(patch_pixel_y / 3) + 1, int(
        patch_pixel_z / 3)
    size_x, size_y, size_z = int(img_x / 3), int(img_y / 3), int(img_z / 3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs * size_x, (xs + 1) * size_x - mask_size_x - 1)
                h = np.random.randint(ys * size_y, (ys + 1) * size_y - mask_size_y - 1)
                z = np.random.randint(zs * size_z, (zs + 1) * size_z - mask_size_z - 1)
                mask[w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0
                loss_mask[:, w:w + mask_size_x, h:h + mask_size_y, z:z + mask_size_z] = 0
    return mask.long(), loss_mask.long()


def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length - 1)
    mask[:, :, z:z + z_length] = 0
    loss_mask[:, :, :, z:z + z_length] = 0
    return mask.long(), loss_mask.long()


def mix_loss(labeled_volume, labeled_label, recoutputs_l, outputs_l, unlabeled_volume, pseudo_label, recoutputs_u,
             outputs_u, l_weight=1.0, u_weight=0.5):
    pseudo_label = pseudo_label.type(torch.int64)
    dice_loss = DICE(outputs_l, labeled_label) * l_weight
    dice_loss += DICE(outputs_u, pseudo_label) * u_weight
    loss_ce = l_weight * F.cross_entropy(outputs_l, labeled_label)
    loss_ce += u_weight * F.cross_entropy(outputs_u, pseudo_label)
    loss_rec = mse(labeled_volume, recoutputs_l)
    loss_rec += mse(unlabeled_volume, recoutputs_u)

    loss = (dice_loss + loss_ce) / 2
    loss_rec = loss_rec / 2
    return loss, dice_loss, loss_ce, loss_rec


def kits_loss(labeled_volume, labeled_label, recoutputs_l, outputs_l, unlabeled_volume, pseudo_label, recoutputs_u,
              outputs_u, l_weight=1.0, u_weight=0.5):
    output_soft_l = F.softmax(outputs_l, dim=1)
    output_soft_u = F.softmax(outputs_u, dim=1)

    dice_loss = tDICE(output_soft_l, labeled_label) * l_weight
    dice_loss += tDICE(output_soft_u, pseudo_label) * u_weight

    loss_ce = l_weight * F.cross_entropy(outputs_l, labeled_label)
    loss_ce += u_weight * F.cross_entropy(outputs_u, pseudo_label)
    loss_rec = mse(labeled_volume, recoutputs_l)
    loss_rec += mse(unlabeled_volume, recoutputs_u)

    loss = (dice_loss + loss_ce) / 2
    loss_rec = loss_rec / 2
    return loss, dice_loss, loss_ce, loss_rec


def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


@torch.no_grad()
def update_ema_teacher_dy(model, ema_model, alpha, iter_num):
    alpha = min(1 - 1 / (iter_num + 1), alpha)
    num = 0
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        num += 1
        if num >= 172:
            break


@torch.no_grad()
def update_ema_teacher(model, ema_model, alpha):
    num = 0
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        num += 1
        if num >= 172:
            break


@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha) / 2) * param1.data).add_(((1 - alpha) / 2) * param2.data)


@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data


class BBoxException(Exception):
    pass


def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))


def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()
