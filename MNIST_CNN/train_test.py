import torch.nn.functional as F
import os
import data_deal
import cnn_model
import torch


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, data_deal.PATH+data_deal.SAVE_NAME)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    model = cnn_model.model
    optimizer = cnn_model.optimizer

    if data_deal.TRAIN_FLAG:
        if os.path.exists(data_deal.PATH):
            checkpoint = torch.load(data_deal.PATH + data_deal.SAVE_NAME)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']

            for now_epoch in range(epoch + 1, epoch + data_deal.EPOCHS + 1):
                train(model, data_deal.DEVICE, data_deal.train_loader, optimizer, now_epoch)
                test(model, data_deal.DEVICE, data_deal.test_loader)

        else:
            os.makedirs(data_deal.PATH)
            for now_epoch in range(1, data_deal.EPOCHS + 1):
                train(model, data_deal.DEVICE, data_deal.train_loader, optimizer, now_epoch)
                test(model, data_deal.DEVICE, data_deal.test_loader)
    else:
        pass


