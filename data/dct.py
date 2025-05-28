import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCT_base_Rec_Module(nn.Module): # 定义一个名为 DCT_base_Rec_Module 的类，继承自 nn.Module
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output] # 输入x的形状为[通道数, 高度, 宽度]，输出形状为[通道数*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]): # 初始化函数
        super().__init__() # 调用父类nn.Module的初始化方法
        
        assert output % window_size == 0 # 断言output必须能被window_size整除
        assert len(level_fliter) > 0 # 断言level_fliter列表的长度必须大于0
        
        self.window_size = window_size # 滑动窗口的大小
        self.grade_N = grade_N # 频率分级的数量
        self.level_N = len(level_fliter) # 频率层级的数量，即level_fliter列表的长度
        self.N = (output // window_size) * (output // window_size) # 计算输出特征图中的patch数量
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False) # 创建一个DCT变换矩阵的参数，不进行梯度更新
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False) # 创建DCT变换矩阵的转置矩阵参数，不进行梯度更新
        
        self.unfold = nn.Unfold( # 定义一个unfold操作，用于将输入图像分割成重叠的patch
            kernel_size=(window_size, window_size), stride=stride # 滑动窗口大小和步长
        )
        self.fold0 = nn.Fold( # 定义一个fold操作，用于将patch合并回图像形式
            output_size=(window_size, window_size), # 输出patch的大小
            kernel_size=(window_size, window_size), # 滑动窗口大小
            stride=window_size # 步长
        )
        
        lm, mh = 2.82, 2 # 定义两个常量，可能用于滤波器设计，但在此处未使用
        level_f = [ # 定义一个包含一个滤波器的列表
            Filter(window_size, 0, window_size * 2) # 创建一个Filter实例，参数为窗口大小，起始频率和终止频率
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter]) # 创建一个ModuleList，包含选定的频率层级滤波器
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)]) # 创建一个ModuleList，包含多个频率分级滤波器
        
        
    def forward(self, x): # 定义前向传播函数
        
        N = self.N # 获取patch数量
        grade_N = self.grade_N # 获取频率分级的数量
        level_N = self.level_N # 获取频率层级的数量
        window_size = self.window_size # 获取滑动窗口的大小
        C, W, H = x.shape # 获取输入张量的形状：通道数，宽度，高度
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)  # 对输入进行unfold操作，先增加一个batch维度，再移除该维度
        

        _, L = x_unfold.shape # 获取unfold后张量的形状，L是patch的数量
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size) # 调整unfold后张量的形状为 [L, C, window_size, window_size]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T # 对每个patch进行DCT变换
        
        y_list = [] # 初始化一个空列表，用于存储经过层级滤波的patch
        for i in range(self.level_N): # 遍历每个频率层级
            x_pass = self.level_filters[i](x_dct) # 使用对应的层级滤波器对DCT系数进行滤波
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch # 对滤波后的DCT系数进行逆DCT变换，得到图像域的patch
            y_list.append(y) # 将处理后的patch添加到列表中
        level_x_unfold = torch.cat(y_list, dim=1) # 将所有层级滤波后的patch在通道维度上拼接起来
        
        grade = torch.zeros(L).to(x.device) # 初始化一个全零张量，用于存储每个patch的频率评分，设备与输入x相同
        w, k = 1, 2 # 初始化权重和权重调整系数
        for _ in range(grade_N): # 遍历每个频率分级
            _x = torch.abs(x_dct) # 取DCT系数的绝对值
            _x = torch.log(_x + 1) # 对绝对值加1后取对数，避免log(0)
            _x = self.grade_filters[_](_x) # 使用对应的分级滤波器对处理后的DCT系数进行滤波
            _x = torch.sum(_x, dim=[1,2,3]) # 在通道、高度、宽度维度上求和，得到每个patch在该分级下的评分
            grade += w * _x            # 将加权后的评分累加到总评分中
            w *= k # 更新权重
        
        _, idx = torch.sort(grade) # 对所有patch的评分进行排序，得到排序后的索引
        max_idx = torch.flip(idx, dims=[0])[:N] # 反转排序索引，取前N个作为评分最高的patch的索引
        maxmax_idx = max_idx[0] # 获取评分最高的patch的索引
        if len(max_idx) == 1: # 如果只有一个最高评分的patch
            maxmax_idx1 = max_idx[0] # 第二高的也设为最高的
        else:
            maxmax_idx1 = max_idx[1] # 否则，获取评分第二高的patch的索引

        min_idx = idx[:N] # 取排序后索引的前N个作为评分最低的patch的索引
        minmin_idx = idx[0] # 获取评分最低的patch的索引
        if len(min_idx) == 1: # 如果只有一个最低评分的patch
            minmin_idx1 = idx[0] # 第二低的也设为最低的
        else:
            minmin_idx1 = idx[1] # 否则，获取评分第二低的patch的索引

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx) # 根据评分最低的索引选择对应的patch
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx) # 根据评分最高的索引选择对应的patch
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1) # 根据评分第二低的索引选择对应的patch
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1) # 根据评分第二高的索引选择对应的patch

        x_minmin = x_minmin.reshape(1, level_N*C*window_size* window_size).transpose(0, 1) # 重塑最低分patch的形状以适应fold操作
        x_maxmax = x_maxmax.reshape(1, level_N*C*window_size* window_size).transpose(0, 1) # 重塑最高分patch的形状以适应fold操作
        x_minmin1 = x_minmin1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1) # 重塑第二低分patch的形状以适应fold操作
        x_maxmax1 = x_maxmax1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1) # 重塑第二高分patch的形状以适应fold操作

        x_minmin = self.fold0(x_minmin) # 使用fold操作将最低分patch转换回图像块形式
        x_maxmax = self.fold0(x_maxmax) # 使用fold操作将最高分patch转换回图像块形式
        x_minmin1 = self.fold0(x_minmin1) # 使用fold操作将第二低分patch转换回图像块形式
        x_maxmax1 = self.fold0(x_maxmax1) # 使用fold操作将第二高分patch转换回图像块形式

       
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1 # 返回四个选定的图像块


        


