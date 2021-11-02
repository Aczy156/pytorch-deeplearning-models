"""
torch test code
"""

# import torch

# # x = torch.Tensor(5, 3)
# x = torch.rand(5, 3)
# print(x)


"""
torch 实现RNN
"""
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

'''
超参数设定
'''
plt.figure(figsize=(8, 5))  # 画布的大小
num = 20  # 每个batch中数据的个数

'''
生成数据
'''
time_steps = np.linspace(0, np.pi, num + 1)
data = np.sin(time_steps)
data = data.reshape((num + 1, 1))
# 因为是监督数据，所以要划分X y
X = data[0: num, :]
y = data[1: num + 1, :]
# 利用matplotlib绘制生成的数据
plt.plot(time_steps[1:num + 1], X, 'r.', label='input_x')
plt.plot(time_steps[1:num + 1], y, 'b.', label='output_y')
plt.legend(loc='best')
plt.show()

'''
手写RNN模型
'''


class myRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(myRNN, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层节点个数
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)  # rnn层
        self.fc = nn.Linear(hidden_dim, output_size)  # 全连接层

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        # 生成预测值和隐状态，预测值传向下一层，隐状态作为记忆参与下一次输入
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)

        return output, hidden


'''
实例化模型
'''
# 指定超参数
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

'''
构建模型
'''
# 初始化手写的RNN网络
rnn = myRNN(input_size, output_size, hidden_dim, n_layers)
# 设置优化器、学习率、损失函数等，用来在训练的过程中更快的逼近最优解
loss = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

'''
手写模型训练函数
'''


def train(rnn, n_steps, print_every):
    # 记忆初始化
    hidden = None
    loss_list = []
    for batch_i, step in enumerate(range(n_steps)):
        optimizer.zero_grad()  # 梯度清零
        # 生成训练数据
        time_steps = np.linspace(step * np.pi, (step + 1) * np.pi, num + 1)
        data = np.sin(time_steps)
        data = data.reshape((num + 1, 1))

        x = data[0:num, :]
        y = data[1:num + 1, :]

        x_tensor = torch.from_numpy(x).unsqueeze(0).type('torch.FloatTensor')
        y_tensor = torch.from_numpy(y).type('torch.FloatTensor')

        prediction, hidden = rnn(x_tensor, hidden)
        hidden = hidden.data
        loss_rate = loss(prediction, y_tensor)
        loss_rate.backward()  # 误差反向传播
        optimizer.step()  # 梯度更新
        loss_list.append(loss_rate)

        if batch_i % print_every == 0:
            plt.plot(time_steps[1:num + 1], x, 'r.', label='input')
            plt.plot(time_steps[1:num + 1], prediction.data.numpy().flatten(), 'b.', label='predict')
            plt.show()

    x = np.linspace(0, n_steps, n_steps)
    plt.plot(x, loss_list, color='blue', linewidth=1.0, linestyle='-', label='loss')
    plt.legend(loc='upper right')
    plt.show()


'''
训练模型并输出
'''
n_steps = 100
print_every = 25
trained_rnn = train(rnn, n_steps, print_every)
