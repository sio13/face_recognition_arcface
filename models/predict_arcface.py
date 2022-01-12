import cv2
import numpy as np
import torch

from .iresnet import i_resnet50


@torch.no_grad()
def encode_sample(wht, img_path=None, img=None):
    wht = torch.load(wht, map_location=torch.device('cpu'))
    img_path = cv2.imread(img_path) or img

    img_path = cv2.resize(img_path, (112, 112))
    img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = np.transpose(img_path, (2, 0, 1))
    img_path = torch.from_numpy(img_path).unsqueeze(0).float()
    img_path.div_(255).sub_(0.5).div_(0.5)
    net = i_resnet50(fp16=False)

    net.load_state_dict(wht)
    net.eval()
    feat = net(img_path).numpy()
    return feat
