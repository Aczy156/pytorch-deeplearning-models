import numpy as np
import matplotlib.pyplot as plt
import config.pytorch_rnn_config as rnn_data_config


def get_pytorch_rnn_data(start):
    time_steps = np.linspace(start * np.pi, (start + 1) * np.pi, rnn_data_config.num + 1)
    data = np.sin(time_steps)
    data = data.reshape((rnn_data_config.num + 1, 1))
    # 因为是监督数据，所以要划分X y
    X = data[0: rnn_data_config.num, :]
    y = data[1: rnn_data_config.num + 1, :]
    plot_data(time_steps, X, y)
    return X, y


def plot_data(time_steps, X, y):
    plt.figure(figsize=(8, 5))
    # 利用matplotlib绘制生成的数据
    plt.plot(time_steps[1:rnn_data_config.num + 1], X, 'r.', label='input_x')
    plt.plot(time_steps[1:rnn_data_config.num + 1], y, 'b.', label='output_y')
    plt.legend(loc='best')
    plt.show()
