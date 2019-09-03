import torch


x = torch.tensor([[1, 2, 3],
                  [1, 2, 4]])

print(x.reshape(-1, 1, 6))
