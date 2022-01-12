from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.arc_margin import ArcMarginProduct
from models.face_loss import ArcFace
from models.iresnet import i_resnet50
from preprocess.random_dataset import SyntheticDataset


def train(train_dataset_,
          batch_size=16,
          shuffle=True,
          log_freq=100,
          save_freq=10,
          step_size=10,
          max_epochs=50,
          num_classes=5749,
          s=64,
          m=0.5,
          num_features=512,
          dropout=0.0,
          lr=1e-1,
          weight_decay=5e-4,
          gamma=0.1):
    def save_model(model_, save_path, name, iter_cnt):
        save_name = os.path.join(save_path, f"{name} {iter_cnt}.pth")
        torch.save(model_.state_dict(), save_name)
        return save_name

    train_loader = DataLoader(train_dataset_,
                              batch_size=batch_size,
                              shuffle=shuffle)

    criterion = ArcFace()

    model = i_resnet50(dropout=dropout, fp16=True, num_features=num_features)
    metric_fc = ArcMarginProduct(num_features, num_classes, s=s, m=m)

    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    start = time.time()
    for i in range(max_epochs):
        scheduler.step()

        model.train()
        for ii, data_ in enumerate(train_loader):
            data_input, label = data_
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            iterations = i * len(train_loader) + ii

            if iterations % log_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = log_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('Training: {}, Epoch: {}, Iter: {}, Iters/s: {}, Loss: {} Acc {}'.format(time_str,
                                                                                               i, ii, speed,
                                                                                               loss[0].sum(),
                                                                                               acc))
                start = time.time()

        if i % save_freq == 0 or i == max_epochs:
            save_model(model, ".", "model", i)

        model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train resnet with ArcFace using specified dataset.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=5749)
    parser.add_argument('--num_features', type=int, default=512)

    args = parser.parse_args()

    train_dataset = SyntheticDataset()
    train(train_dataset_=train_dataset,
          batch_size=args.batch_size,
          max_epochs=args.max_epochs,
          step_size=args.step_size,
          num_classes=args.num_classes,
          num_features=args.num_features)
