import data_deal
import model_lstm
import torch
import os
import matplotlib.pyplot as plt
import numpy as np


def train(model, device, train_x, train_y, optimizer, now_epoch, epoch):
    model.train()
    train_x, train_y = train_x.to(device), train_y.to(device)
    for i in range(now_epoch+1, now_epoch+epoch+1):
        out = model(train_x)
        loss = model_lstm.loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    torch.save({
        'epoch': now_epoch+epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, data_deal.PATH + data_deal.SAVE_NAME)


def test(model, device, dataset_x):
    model.eval()
    dataset_x = dataset_x.to(device)
    pred_test = model(dataset_x).view(-1).data.cpu().numpy()
    pred_test = np.concatenate((np.zeros(data_deal.DAYS_FOR_TRAIN), pred_test))
    assert len(pred_test) == len(data_deal.data_close)

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_deal.data_close, 'b', label='real')
    plt.plot((data_deal.train_size, data_deal.train_size), (0, 1), 'g--')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    model = model_lstm.model
    optimizer = model_lstm.optimizer
    train_flag = False

    if train_flag:
        if os.path.exists(data_deal.PATH):
            checkpoint = torch.load(data_deal.PATH + data_deal.SAVE_NAME)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            train(model, data_deal.DEVICE, data_deal.train_x, data_deal.train_y, optimizer, epoch, data_deal.EPOCHS)
            test(model, data_deal.DEVICE, data_deal.dataset_x)
        else:
            os.makedirs(data_deal.PATH)
            train(model, data_deal.DEVICE, data_deal.train_x, data_deal.train_y, optimizer, 0, data_deal.EPOCHS)
    else:
        checkpoint = torch.load(data_deal.PATH + data_deal.SAVE_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        test(model, data_deal.DEVICE, data_deal.dataset_x)

