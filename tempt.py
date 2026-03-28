import torch
import torch.nn as nn
from loss import AspectSentimentLoss

ce = nn.CrossEntropyLoss(ignore_index=-100)

x = torch.arange(12)

y = x.view(2, 2, 3)

for i in range(2):
    for j in range(2):
        for k in range(3):
            print(y[i][j][k])


z = x.view(4, 3)

for i in range(4):
    for j in range(3):
        print(z[i][j])
