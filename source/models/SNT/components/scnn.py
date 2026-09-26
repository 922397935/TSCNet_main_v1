import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialConv(nn.Module):
    """
    简单的单纯复形卷积层（以 0-1 阶邻接为例）
    inputs:
        x: [batch, node_num, feature_dim]
        lap_0: 节点级别拉普拉斯矩阵
        lap_1: 边级别拉普拉斯矩阵
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_0 = nn.Linear(in_channels, out_channels)
        self.linear_1 = nn.Linear(in_channels, out_channels)

    def forward(self, x, lap_0, lap_1):
        # x 是 0-单形特征（节点特征）
        h0 = self.linear_0(torch.matmul(lap_0, x))
        h1 = self.linear_1(torch.matmul(lap_1, x))
        return F.leaky_relu(h0 + h1)
