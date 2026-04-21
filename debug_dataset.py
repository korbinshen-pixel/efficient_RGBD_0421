"""调试数据集加载"""
import yaml
from pathlib import Path

data_dir = Path('/home/usr2/data_trainer_ws/efficinet20260307/20260307_155324_01')

# 1. 检查 train.txt
with open(data_dir / 'train.txt') as f:
    image_ids = [line.strip() for line in f]

print(f"train.txt 前 5 个 ID:")
for img_id in image_ids[:5]:
    print(f"  '{img_id}' (type: {type(img_id).__name__})")

# 2. 检查 gt.yml
with open(data_dir / 'gt.yml') as f:
    gt_dict = yaml.safe_load(f)

print(f"\ngt.yml 前 5 个键:")
for key in list(gt_dict.keys())[:5]:
    print(f"  {repr(key)} (type: {type(key).__name__})")
    print(f"    值类型: {type(gt_dict[key])}")
    if isinstance(gt_dict[key], list) and len(gt_dict[key]) > 0:
        print(f"    第一个元素: {list(gt_dict[key][0].keys())}")

# 3. 检查 info.yml
with open(data_dir / 'info.yml') as f:
    info_dict = yaml.safe_load(f)

print(f"\ninfo.yml 前 5 个键:")
for key in list(info_dict.keys())[:5]:
    print(f"  {repr(key)} (type: {type(key).__name__})")

# 4. 测试键匹配
test_id = image_ids[0]
print(f"\n测试第一个图像 ID: '{test_id}'")
print(f"  在 gt_dict 中? {test_id in gt_dict}")
print(f"  int(test_id) 在 gt_dict 中? {int(test_id) in gt_dict}")
print(f"  在 info_dict 中? {test_id in info_dict}")
print(f"  int(test_id) 在 info_dict 中? {int(test_id) in info_dict}")
