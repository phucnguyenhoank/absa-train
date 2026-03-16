import torch

l = [torch.rand(size=(2, 3)) for _ in range(4)]
o = torch.stack(l, dim=1)
print(o.shape)
