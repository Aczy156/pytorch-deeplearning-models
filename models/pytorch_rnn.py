import torch.nn as nn
import config.pytorch_rnn_config as rnn_model_config


class myRNN(nn.Module):
    def __init__(self):
        super(myRNN, self).__init__()
        """
        初始化模型参数
        a) 模型中隐藏层的节点个数（因为是RNN，也就是循环的个数）
        b) 每个循环的单元的输入、输出、层数
        """
        self.input_size = rnn_model_config.input_size
        self.output_size = rnn_model_config.output_size
        self.hidden_dim = rnn_model_config.hidden_dim  # 隐藏层节点个数
        self.n_layers = rnn_model_config.n_layers
        self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.n_layers, batch_first=True)  # rnn层
        self.fc = nn.Linear(self.hidden_dim, self.output_size)  # 全连接层

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        # 生成预测值和隐状态，预测值传向下一层，隐状态作为记忆参与下一次输入
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)

        return output, hidden
