import torch
from torchvision import datasets, transforms


BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.0381,))
                   ]))
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor,
                       transforms.Normalize((0.1307,), (0.0381,))
                   ]))
)
