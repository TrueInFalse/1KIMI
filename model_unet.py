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
    in_channels: int = 3
) -> None:
    """加载本地ResNet34预训练权重。
    
    打印详细的权重匹配统计信息。
    
    Args:
        model: smp.Unet模型
        weights_path: 本地权重文件路径
        in_channels: 输入通道数（当前统一为3）
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
    
    # 统计信息
    matched_layers = []
    mismatched_layers = []
    skipped_fc = []
    missing_in_model = []
    missing_in_ckpt = []
    
    # 分析所有checkpoint中的权重
    for name, param in state_dict.items():
        # 跳过全连接层（分类头）
        if 'fc' in name:
            skipped_fc.append(name)
            continue
        
        if name in encoder_state:
            if encoder_state[name].shape == param.shape:
                matched_layers.append((name, param.shape))
            else:
                mismatched_layers.append((name, param.shape, encoder_state[name].shape))
        else:
            missing_in_model.append(name)
    
    # 检查模型中有但checkpoint中没有的层
    ckpt_keys = set(k for k in state_dict.keys() if 'fc' not in k)
    model_keys = set(encoder_state.keys())
    for name in model_keys - ckpt_keys:
        missing_in_ckpt.append(name)
    
    # 打印统计信息
    print(f'\n{"="*60}')
    print(f'[预训练权重加载报告] {weights_path.name}')
    print(f'{"="*60}')
    print(f'模型encoder层数: {len(encoder_state)}')
    print(f'Checkpoint层数: {len(state_dict)} (含{len(skipped_fc)}层FC被跳过)')
    print(f'✅ 匹配并成功加载: {len(matched_layers)} 层')
    
    if mismatched_layers:
        print(f'⚠️  形状不匹配跳过: {len(mismatched_layers)} 层')
        for name, ckpt_shape, model_shape in mismatched_layers[:3]:
            print(f'   - {name}: checkpoint{list(ckpt_shape)} vs model{list(model_shape)}')
    
    if missing_in_model:
        print(f'⚠️  Checkpoint中有但模型中无: {len(missing_in_model)} 层')
        for name in missing_in_model[:3]:
            print(f'   - {name}')
    
    if missing_in_ckpt:
        print(f'⚠️  模型中有但Checkpoint中无: {len(missing_in_ckpt)} 层')
        for name in missing_in_ckpt[:3]:
            print(f'   - {name}')
    
    # 执行加载
    for name, param in state_dict.items():
        if 'fc' in name:
            continue
        if name in encoder_state and encoder_state[name].shape == param.shape:
            encoder_state[name].copy_(param)
    
    print(f'[结论] 成功加载 {len(matched_layers)}/{len(encoder_state)} 层预训练权重')
    print(f'{"="*60}\n')


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
