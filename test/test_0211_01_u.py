# import numpy as np
# import cripser
# import time

# # 模拟 DRIVE 数据集尺寸（缩放到 128x128 测试）
# img_size = 128
# vessel_prob = np.random.rand(img_size, img_size).astype(np.float32)

# # 转换为 cripser 需要的格式（关键步骤！）
# vessel_uint8 = (vessel_prob * 255).astype(np.uint8)

# # 测试计算速度（这是 AutoDL GPU 服务器上应该关心的）
# start = time.time()
# persistence = cripser.computePH(vessel_uint8, maxdim=1)
# elapsed = time.time() - start

# print(f"处理 {img_size}x{img_size} 图像耗时: {elapsed:.3f}s")
# print(f"0维连通分量数: {np.sum(persistence[:, 0] == 0)}")
# print(f"1维环路数: {np.sum(persistence[:, 0] == 1)}")

# if elapsed < 1.0:
#     print("✓ 速度正常，可以集成到训练循环")
# else:
#     print("⚠ 速度较慢，建议使用 TTA 策略而非每 batch 计算")

import numpy as np
# 假设用A
from cubical_ripser import cubical_ripser

# 测试2D输入
img = np.random.rand(128, 128)
# 计算0维持续图
result = cubical_ripser(img, dim=0)
print(result)  # 期望输出: [(birth, death), ...]