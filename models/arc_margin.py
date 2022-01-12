import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_, label_):
        cosine_ = F.linear(F.normalize(input_), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine_, 2)).clamp(0, 1))
        phi = cosine_ * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_ > self.th, phi, cosine_ - self.mm)
        one_hot = torch.zeros(cosine_.size())
        one_hot.scatter_(1, label_.view(-1, 1).long(), 1)
        output_ = (one_hot * phi) + ((1.0 - one_hot) * cosine_)
        output_ *= self.s

        return output_
