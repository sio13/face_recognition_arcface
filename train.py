from __future__ import print_function

import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from models.arc_margin import ArcMarginProduct
from models.face_loss import ArcFace
from models.iresnet import get_model


def train():
    train_dataset = SyntheticDataset()
    trainloader = DataLoader(train_dataset,
                             batch_size=16,
                             shuffle=True)

    print('{} train iters per epoch:'.format(len(trainloader)))

    criterion = ArcFace()

    model = get_model("r50", dropout=0.0, fp16=True, num_features=512)
    metric_fc = ArcMarginProduct(512, 5749, s=30, m=0.5)

    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=1e-1, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    start = time.time()
    for i in range(50):
        scheduler.step()

        model.train()
        for ii, data_ in enumerate(trainloader):
            data_input, label = data_
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % 100 == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = 100 / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str,
                                                                                   i, ii, speed, loss[0].sum(),
                                                                                   acc))
                start = time.time()

        if i % 10 == 0 or i == 50:
            save_model(model, ".", "model", i)

        model.eval()


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def save_model(model_, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, f"{name} {iter_cnt}.pth")
    torch.save(model_.state_dict(), save_name)
    return save_name


if __name__ == '__main__':
    train()
