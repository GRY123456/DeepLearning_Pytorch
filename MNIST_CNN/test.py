import torch


x = torch.tensor([[1, 9, 3],
                  [4, 5, 6]])
print(x)
print(x.max(1, keepdim=True))
print(x.max(1, keepdim=True)[1])
