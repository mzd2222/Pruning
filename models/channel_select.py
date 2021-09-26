import numpy as np
import torch
import torch.nn as nn

'''
该层中的indexes存储的是对应的bn或者卷积层的要裁剪的通道                                                  
该层参与前向传播,但是只是简单的将各通道的输出与indexes相乘
不参与反向传播,反向传播时直接跳过该层
indexes的值为0/1
用于对bn或者卷积层剩余的通道
'''

class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor, activations=None):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output