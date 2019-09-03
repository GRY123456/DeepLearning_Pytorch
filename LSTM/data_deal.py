import numpy as np
import tushare as ts
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
PATH = "./model_save/"
SAVE_NAME = "model_data.tar"

data_close = ts.get_k_data('000001', start='2018-01-01', index=True)['close'].values  # 获取上证指数
data_close = data_close.astype('float32')  # 转换数据类型

max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

DAYS_FOR_TRAIN = 10


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    dataset_x, dataset_y = [], []
    for i in range(len(data)-days_for_train):
        _x = data[i:(i+days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i+days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

train_size = int(len(dataset_x) * 0.7)
train_x = dataset_x[:train_size]
train_y = dataset_y[:train_size]

train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
train_y = train_y.reshape(-1, 1, 1)

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
dataset_x = torch.from_numpy(dataset_x)
