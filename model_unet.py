# -*- coding: utf-8 -*-
"""
文件名: model_unet.py
项目: retina_ph_seg
功能: U-Net模型定义，基于segmentation-models-pytorch
作者: AI Assistant
创建日期: 2026-02-07
Python版本: 3.12+

依赖关系:
  - 上游: config.yaml（配置读取）
  - 下游: train_baseline.py（训练）, evaluate.py（评估）

主要类/函数:
  - get_unet_model(): 创建smp.Unet模型
  - load_model(): 加载已保存的模型

更新记录:
  - v1.0: 初始版本，使用smp.Unet + ResNet34编码器
  - v1.1: 添加本地权重加载功能，绕开网络请求
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
import yaml


def load_local_resnet34_weights(
    model: nn.Module,
    weights_path: str = './pretrained_weights/resnet34-333f7ec4.pth',
    in_channels: int = 1
) -> None:
    """加载本地ResNet34预训练权重。
    
    处理1通道输入的情况：将3通道权重平均到1通道。
    
    Args:
        model: smp.Unet模型
        weights_path: 本地权重文件路径
        in_channels: 输入通道数
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f'警告: 本地权重文件不存在: {weights_path}')
        print('模型将使用随机初始化权重')
        return
    
    # 加载torchvision格式的ResNet34权重
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # 获取模型编码器的state_dict
    encoder_state = model.encoder.state_dict()
    
    # 过滤并转换权重
    # torchvision的ResNet34与smp的encoder名称可能不同，需要映射
    loaded_layers = 0
    for name, param in state_dict.items():
        # 跳过全连接层（分类头）
        if 'fc' in name:
            continue
        
        # 处理第一层卷积（conv1）的通道数不匹配
        if name == 'conv1.weight' and in_channels == 1:
            # 将3通道权重平均到1通道: [64, 3, 7, 7] -> [64, 1, 7, 7]
            param = param.mean(dim=1, keepdim=True)
        
        # smp的encoder通常使用resnet34.encoder.conv1的命名
        # 需要将torchvision的命名转换为smp的命名
        encoder_name = name
        if encoder_name in encoder_state:
            if encoder_state[encoder_name].shape == param.shape:
                encoder_state[encoder_name].copy_(param)
                loaded_layers += 1
    
    print(f'成功加载 {loaded_layers} 层本地预训练权重')


def get_unet_model(
    in_channels: int = 1,
    encoder: str = 'resnet34',
    pretrained: bool = True,
    activation: Optional[str] = None,
    local_weights_path: str = './pretrained_weights/resnet34-333f7ec4.pth'
) -> nn.Module:
    """创建U-Net模型。
    
    使用segmentation-models-pytorch的Unet实现，ResNet34编码器。
    支持本地预训练权重加载，完全绕开网络请求。
    输出为logits（无sigmoid），配合smp.losses.DiceLoss使用。
    
    Args:
        in_channels: 输入通道数，1=绿色通道，3=RGB
        encoder: 编码器名称，默认resnet34
        pretrained: 是否使用ImageNet预训练权重
        activation: 输出激活函数，None表示输出logits
        local_weights_path: 本地权重文件路径
        
    Returns:
        model: smp.Unet模型实例
        
    Example:
        >>> model = get_unet_model(in_channels=1)
        >>> x = torch.randn(2, 1, 512, 512)
        >>> y = model(x)  # [2, 1, 512, 512]
    """
    # 先创建模型（不加载权重，避免网络请求）
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,  # 不自动下载权重
        in_channels=in_channels,
        classes=1,              # 二分类：血管/背景
        activation=activation,  # 输出logits，不激活
    )
    
    # 手动加载本地权重
    if pretrained:
        print(f'尝试加载本地权重: {local_weights_path}')
        load_local_resnet34_weights(model, local_weights_path, in_channels)
    
    return model


def load_model(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda',
    local_weights_path: str = './pretrained_weights/resnet34-333f7ec4.pth'
) -> nn.Module:
    """从检查点加载模型。
    
    Args:
        checkpoint_path: 模型权重文件路径
        device: 加载设备
        local_weights_path: 本地预训练权重路径（用于创建模型结构）
        
    Returns:
        model: 加载权重后的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 尝试从checkpoint获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = get_unet_model(
            in_channels=config.get('in_channels', 1),
            encoder=config.get('encoder', 'resnet34'),
            pretrained=False,  # 加载权重时不需要预训练
            local_weights_path=local_weights_path
        )
    else:
        # 使用默认配置
        model = get_unet_model(in_channels=1, pretrained=False, local_weights_path=local_weights_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量。
    
    Args:
        model: PyTorch模型
        
    Returns:
        num_params: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(model: nn.Module) -> None:
    """冻结编码器（用于微调场景）。
    
    Args:
        model: smp.Unet模型
    """
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """解冻编码器。
    
    Args:
        model: smp.Unet模型
    """
    for param in model.encoder.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    # 简单测试
    print('测试U-Net模型...')
    
    model = get_unet_model(in_channels=1, encoder='resnet34', pretrained=False)
    print(f'模型参数量: {count_parameters(model):,}')
    
    # 前向传播测试
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    print(f'输出值域: [{y.min():.3f}, {y.max():.3f}]')
    
    print('\n模型创建成功！')
