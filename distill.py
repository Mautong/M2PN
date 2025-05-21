import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def feature_distillation_loss(s_features, t_features):
    """
    计算中间层特征的蒸馏损失
    s_features: 学生网络的中间层特征列表
    t_features: 教师网络的中间层特征列表
    """
    loss = 0
    for sf, tf in zip(s_features, t_features):
        # 调整特征图大小以匹配
        if sf.size() != tf.size():
            sf = F.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
        # 计算特征图的L2距离
        loss += F.mse_loss(sf, tf)
    return loss / len(s_features)


def attention_transfer_loss(s_feature, t_feature):
    """
    计算注意力迁移损失，关注去雨区域
    """
    # 计算空间注意力图
    s_spatial = torch.mean(torch.abs(s_feature), dim=1, keepdim=True)
    t_spatial = torch.mean(torch.abs(t_feature), dim=1, keepdim=True)
    
    # 归一化注意力图
    s_spatial = F.normalize(s_spatial.view(s_spatial.size(0), -1), dim=1).view_as(s_spatial)
    t_spatial = F.normalize(t_spatial.view(t_spatial.size(0), -1), dim=1).view_as(t_spatial)
    
    # 计算通道注意力
    s_channel = torch.mean(s_feature, dim=[2, 3])
    t_channel = torch.mean(t_feature, dim=[2, 3])
    
    # 归一化通道注意力
    s_channel = F.normalize(s_channel, dim=1)
    t_channel = F.normalize(t_channel, dim=1)
    
    # 组合空间和通道注意力损失
    spatial_loss = F.mse_loss(s_spatial, t_spatial)
    channel_loss = F.mse_loss(s_channel, t_channel)
    
    return spatial_loss + channel_loss


def feature_similarity_loss(s_feature, t_feature):
    """
    计算特征相似性损失，专注于去雨相关的特征
    """
    # 确保空间维度匹配
    if s_feature.size()[2:] != t_feature.size()[2:]:
        s_feature = F.interpolate(s_feature, size=t_feature.shape[2:], mode='bilinear', align_corners=False)
    
    # 计算特征图的结构相似性
    s_feat = F.normalize(s_feature.view(s_feature.size(0), -1), dim=1)
    t_feat = F.normalize(t_feature.view(t_feature.size(0), -1), dim=1)
    
    # 计算结构相似性损失
    struct_loss = 1 - F.cosine_similarity(s_feat, t_feat).mean()
    
    # 计算局部一致性损失
    local_loss = F.mse_loss(s_feature, t_feature)
    
    return struct_loss + local_loss


def distillation_loss(y_student, y_teacher, s_features, t_features, input_img, alpha=0.6, beta=0.3, gamma=0.1):
    """
    知识蒸馏损失函数
    Args:
        y_student: PReNet输出的去雨图像
        y_teacher: M2PN输出的去雨图像
        s_features: PReNet的特征列表
        t_features: M2PN的特征列表
        input_img: 输入的雨图像
    """
    # 1. 图像级去雨效果损失
    ssim_criterion = SSIM()
    ssim_value = ssim_criterion(y_student, y_teacher)
    ssim_loss = 1 - ssim_value
    
    # 计算雨区域mask
    with torch.no_grad():
        rain_region = torch.abs(input_img - y_teacher)
        rain_mask = torch.mean(rain_region, dim=1, keepdim=True)
        rain_mask = torch.sigmoid(rain_mask * 5.0)
    
    # 在雨区域加权的L1损失
    l1_loss = F.l1_loss(y_student * (1 + rain_mask), y_teacher * (1 + rain_mask))
    
    # 2. 特征蒸馏损失
    feature_loss = torch.tensor(0.0).to(y_student.device)
    if len(s_features) > 0 and len(t_features) > 0:
        # 使用最后一次迭代的特征
        s_last = s_features[-1]  # [B, 32, H, W]
        t_last = t_features[-1]  # [B, C, H, W]
        
        # 确保空间维度匹配
        if s_last.size()[2:] != t_last.size()[2:]:
            t_last = F.interpolate(t_last, size=s_last.size()[2:], mode='bilinear', align_corners=False)
        
        # 将教师特征转换为与学生特征相同的通道数
        if not hasattr(distillation_loss, 'channel_adapt'):
            distillation_loss.channel_adapt = nn.Conv2d(t_last.size(1), s_last.size(1), 1, bias=False).to(t_last.device)
            # 使用kaiming初始化
            nn.init.kaiming_normal_(distillation_loss.channel_adapt.weight.data)
        
        # 转换教师特征
        t_last = distillation_loss.channel_adapt(t_last)
        
        # 计算注意力图
        def get_attention_maps(feature):
            # 使用L2范数计算注意力
            attention = torch.norm(feature, p=2, dim=1, keepdim=True)
            attention = F.normalize(attention.view(attention.size(0), -1), dim=1).view_as(attention)
            return attention
        
        # 获取注意力图
        s_attention = get_attention_maps(s_last)
        t_attention = get_attention_maps(t_last)
        
        # 计算注意力一致性损失
        attention_loss = F.mse_loss(s_attention, t_attention)
        
        # 在注意力区域计算特征一致性
        s_feat = s_last * s_attention
        t_feat = t_last * t_attention
        
        # 特征一致性损失
        feat_loss = F.mse_loss(s_feat, t_feat)
        
        # 组合特征损失，增加权重
        feature_loss = attention_loss + feat_loss
    
    # 调整损失权重
    total_loss = alpha * ssim_loss + beta * l1_loss + gamma * feature_loss
    
    # 返回损失详情
    loss_details = {
        'ssim_loss': ssim_loss.item(),
        'l1_loss': l1_loss.item(),
        'feature_loss': feature_loss.item()
    }
    
    return total_loss, loss_details
