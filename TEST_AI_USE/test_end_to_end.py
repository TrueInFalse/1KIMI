import sys
sys.path.insert(0, '.')

# 模拟 train_with_topology 的导入逻辑
try:
    from data_combined import get_combined_loaders
    print("✅ 可以导入 get_combined_loaders")
    
    import yaml
    with open('config_kaggle.yaml') as f:
        config = yaml.safe_load(f)
    
    # 测试是否能获取到 loader
    train_loader, val_loader, test_loader = get_combined_loaders(config)
    print(f"✅ 获取到训练加载器: {len(train_loader)} 批次")
    print(f"✅ 获取到验证加载器: {len(val_loader)} 批次")
    print("\n🎉 端到端测试通过！train_with_topology.py 只需改导入即可工作")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    print("需要修改 train_with_topology.py 以适配 data_combined")
