import math
import random
from typing import Tuple
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    output_flipped = torch.flip(output_flipped, dims=[3])
    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp
    return output_flipped


def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'
    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()
    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)
    preds[:, :, 0] = idx % w
    preds[:, :, 1] = torch.floor(idx / w)
    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)
    preds *= pred_mask
    return preds, maxvals


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    assert trans is not None
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25
    preds = coords.clone().cpu().numpy()
    for i in range(coords.shape[0]):
        preds[i] = affine_points(preds[i], trans[i])
    return preds, maxvals.cpu().numpy()


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    xmax = xmin + w
    ymax = ymin + h
    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h
    return xmin, ymin, xmax, ymax


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, mask_topnet=None):
        for t in self.transforms:
            image, target, mask_topnet = t(image, target, mask_topnet)
        return image, target, mask_topnet


class ToTensor(object):
    def __call__(self, image, target, mask_topnet=None):
        image = F.to_tensor(image)
        if mask_topnet is not None:
            mask_topnet = F.to_tensor(mask_topnet)
        return image, target, mask_topnet


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, mask_topnet=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, mask_topnet


class AffineTransform(object):
    def __init__(self,
                 scale: Tuple[float, float] = None,
                 rotation: Tuple[int, int] = None,
                 fixed_size: Tuple[int, int] = (256, 192)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target, mask_topnet=None):
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])
        src_p3 = src_center + np.array([src_w / 2, 0])
        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])
        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])
            src_p3 = src_center + np.array([src_w / 2, 0])
        if self.rotation is not None:
            angle = random.randint(*self.rotation)
            angle = angle / 180 * math.pi
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])
        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)
        trans = cv2.getAffineTransform(src, dst)
        dst /= 4
        reverse_trans = cv2.getAffineTransform(dst, src)
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),
                                    flags=cv2.INTER_LINEAR)
        if mask_topnet is not None:
            mask_topnet = cv2.warpAffine(mask_topnet,
                                        trans,
                                        tuple(self.fixed_size[::-1]),
                                        flags=cv2.INTER_LINEAR)
        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            target["keypoints"] = kps
        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target, mask_topnet
