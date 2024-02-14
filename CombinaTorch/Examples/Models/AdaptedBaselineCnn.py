import math

import torch.nn.functional as F
from torch import nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class BaselineCnn(nn.Module):
    def __init__(self,
                 # classes_num,
                 # hidden_size,  # 64
                 n_hidden,
                 task_list,
                 drop_rate
                 ):
        super(BaselineCnn, self).__init__()
        self.name = 'baselinecnn'
        self.task_list = [t.output_module for t in task_list]
        self.hidden = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()
        self.drop_rate = drop_rate

        for k in range(n_hidden):
            self.hidden.append(
                nn.Conv2d(in_channels=1 if k == 0 else pow(2, 5 + k),
                          out_channels=pow(2, 6 + k),
                          kernel_size=(5, 5), stride=(2, 2),
                          padding=(2, 2), bias=False)
            )
            self.hidden_bn.append(
                nn.BatchNorm2d(pow(2, 6 + k))
            )

        self.task_nets = nn.ModuleList()
        for t in task_list:
            self.task_nets.append(
                nn.Linear(pow(2, 5 + n_hidden), len(t.output_labels))
            )

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
        #                        kernel_size=(5, 5), stride=(2, 2),
        #                        padding=(2, 2), bias=False)
        #
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
        #                        kernel_size=(5, 5), stride=(2, 2),
        #                        padding=(2, 2), bias=False)
        #
        # self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
        #                        kernel_size=(5, 5), stride=(2, 2),
        #                        padding=(2, 2), bias=False)
        #
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
        #                        kernel_size=(5, 5), stride=(2, 2),
        #                        padding=(2, 2), bias=False)

        # self.fc1 = nn.Linear(512, classes_num, bias=True)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        for i in range(len(self.hidden)):
            init_layer(self.hidden[i])
        for i in range(len(self.task_nets)):
            init_layer(self.task_nets[i])
        for i in range(len(self.hidden_bn)):
            init_bn(self.hidden_bn[i])
        # init_layer(self.conv1)
        # init_layer(self.conv2)
        # init_layer(self.conv3)
        # init_layer(self.conv4)
        # init_layer(self.fc1)

        # init_bn(self.bn1)
        # init_bn(self.bn2)
        # init_bn(self.bn3)
        # init_bn(self.bn4)

    def activate(self, x, activation):
        if activation == 'softmax':
            x = F.log_softmax(x, dim=-1)
            return x
        elif activation == 'sigmoid':
            x = F.sigmoid(x)
            return x

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        for h_id in range(len(self.hidden)):
            x = F.relu(self.hidden_bn[h_id](self.hidden[h_id](x)))
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        # x = F.log_softmax(self.fc1(x), dim=-1)

        return tuple(
            self.activate(self.task_nets[t_id](x), self.task_list[t_id]) for t_id in range(len(self.task_nets)))
