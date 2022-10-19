import torch

a = torch.tensor([True, False, False, True])

b = torch.sum(a)

print(a.size(dim=0))