import numpy as np
import matplotlib.pyplot as plt
from data_preparation.pytorch_rnn_data_preparation import get_pytorch_rnn_data
from models.pytorch_rnn import myRNN
import config.pytorch_rnn_config as rnn_train_config
import torch.nn as nn
import torch


def train(rnn, n_steps, print_every):
    # 记忆初始化
    hidden = None
    loss_list = []
    for batch_i, start in enumerate(range(n_steps)):
        optimizer.zero_grad()  # 梯度清零
        # 生成训练数据
        x, y = get_pytorch_rnn_data(start)

        x_tensor = torch.from_numpy(x).unsqueeze(0).type('torch.FloatTensor')
        y_tensor = torch.from_numpy(y).type('torch.FloatTensor')

        prediction, hidden = rnn(x_tensor, hidden)
        hidden = hidden.data
        loss_rate = loss(prediction, y_tensor)
        loss_rate.backward()  # 误差反向传播
        optimizer.step()  # 梯度更新
        loss_list.append(loss_rate)

        # if batch_i % print_every == 0:
        #     plt.plot(time_steps[1:num + 1], x, 'r.', label='input')
        #     plt.plot(time_steps[1:num + 1], prediction.data.numpy().flatten(), 'b.', label='predict')
        #     plt.show()

    x = np.linspace(0, rnn_train_config.n_steps, rnn_train_config.n_steps)
    plt.figure(figsize=(8, 5))
    plt.plot(x, loss_list, color='blue', linewidth=1.0, linestyle='-', label='loss')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    X, y = get_pytorch_rnn_data(0)  # 获取数据

    rnn = myRNN()  # 获取模型
    loss = nn.MSELoss()  # 设置损失函数
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)  # 优化器

    train(rnn, rnn_train_config.n_steps, rnn_train_config.print_every)
