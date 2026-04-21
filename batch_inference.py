"""批量推理并评估"""
import sys
from pathlib import Path
from inference import inference
from config import Config

config = Config()
data_dir = Path('20260307_155324_01')
rgb_dir = data_dir / 'rgb'

# 读取测试集
with open(data_dir / 'test.txt') as f:
    test_ids = [line.strip() for line in f]

print(f"📦 测试集大小: {len(test_ids)} 张")

# 随机选 5 张测试
import random
random.seed(42)
sample_ids = random.sample(test_ids, min(5, len(test_ids)))

for img_id in sample_ids:
    image_path = rgb_dir / f'{img_id}.png'
    print(f"\n{'='*60}")
    print(f"测试图像: {image_path}")
    
    try:
        inference(str(image_path), 
                 f'{config.checkpoints_dir}/best_model_phi0.pth', 
                 config)
    except Exception as e:
        print(f"❌ 推理失败: {e}")

print(f"\n{'='*60}")
print("✅ 批量测试完成")
