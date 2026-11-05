import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .unet_vgg import UNet_VGG


class UNetPred(object):
    def __init__(self, weights_path):
        self.net = UNet_VGG()
        self.input_shape = self.net.input_shape
        if not torch.cuda.is_available():
            raise RuntimeError("UNet 分割需要 CUDA 设备")
        self.cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def resize_image(self, image, target_shape):
        h, w = image.shape[:2]
        target_h, target_w = target_shape
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        new_image[top:top + nh, left:left + nw] = resized
        return new_image, nw, nh

    def get_mask(self, image):
        orininal_h, orininal_w = image.shape[:2]
        image_data, nw, nh = self.resize_image(image, self.input_shape)
        image_data = image_data.astype(np.float32) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        return pr.astype(np.uint8)
