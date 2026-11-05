import numpy as np
import torch
from .hrnet import HighResolutionNet
from . import transforms


class HRNetPred(object):
    def __init__(self, weights_path, mode):
        if mode != "add_mask":
            raise ValueError("仅支持 add_mask 权重与四通道输入")
        self.mode = mode
        self.input_channel = 4 if self.mode == "add_mask" else 3
        self.model = HighResolutionNet(base_channel=32, num_joints=11, input_channel=self.input_channel)
        self.model = torch.nn.DataParallel(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = torch.load(weights_path, map_location=self.device)
        weights = weights if "model" not in weights else weights["model"]
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()
        self.resize_hw = (256, 192)
        # RGB 使用 ImageNet 归一化；类别掩膜经 ToTensor /255 后作为第 4 通道。
        self.data_transform = transforms.Compose([
            transforms.AffineTransform(scale=(1, 1), fixed_size=self.resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    def predict(self, image, mask, flip_test=True):
        if self.mode == "add_mask":
            img_tensor, target, mask = self.data_transform(image, {"box": [0, 0, image.shape[1] - 1, image.shape[0] - 1]}, mask)
            img_tensor = torch.cat((img_tensor, mask), dim=0)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
        with torch.inference_mode():
            outputs = self.model(img_tensor.to(self.device))
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(self.model(flip_tensor.to(self.device)), self.flip_pairs),
                )
                # 翻转热图向右补偿一像素，再与原预测平均。
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5
            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
        return keypoints, scores
