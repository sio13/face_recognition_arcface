import cv2
import numpy as np
import torch

from .resnet50 import iresnet50


@torch.no_grad()
def encode_sample(wht, img_path):
    img_path = cv2.imread(img_path)

    img_path = cv2.resize(img_path, (112, 112))
    img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = np.transpose(img_path, (2, 0, 1))
    img_path = torch.from_numpy(img_path).unsqueeze(0).float()
    img_path.div_(255).sub_(0.5).div_(0.5)
    net = iresnet50(False, fp16=False)
    net.load_state_dict(wht)
    net.eval()
    feat = net(img_path).numpy()
    return feat

