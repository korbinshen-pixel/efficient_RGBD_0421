"""
benchmark_inference.py

测试 EfficientPose 推理速度：
- 读取一个真实样本，测单次推理延迟
- 用随机输入测 batch 推理吞吐量（FPS）

运行：
    python benchmark_inference.py
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from models.efficientpose import EfficientPose
from datasets.pallet_dataset import PalletDataset, collate_fn


@torch.no_grad()
def benchmark():
    config = Config()
    device = torch.device(config.device)

    # 1. 准备数据集和 DataLoader（只用于拿一个真实尺寸的样本）
    test_dataset = PalletDataset(
        dataset_path=config.dataset_path,
        object_dir=config.object_dir,
        split='test',
        transform=None,
        config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 2. 构建模型（注意 in_channels 根据是否有深度图自动切换）
    in_ch = 4 if getattr(test_dataset, "use_depth", False) else 3
    model = EfficientPose(phi=config.phi,
                          num_classes=config.num_classes,
                          in_channels=in_ch).to(device)
    model.eval()

    # 3. 加载权重（和你 eval.py 的逻辑一致）
    ckpt_dir = Path(config.checkpoints_dir)
    model_files = sorted(ckpt_dir.glob("best_model_phi*.pth"))
    if not model_files:
        raise FileNotFoundError(f"在 {ckpt_dir} 中未找到 best_model_phi*.pth")
    ckpt_path = model_files[0]
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"加载模型: {ckpt_path}")

    if isinstance(ckpt, dict):
        key = next((k for k in ["model_state_dict", "model", "state_dict"]
                    if k in ckpt), None)
        state_dict = ckpt[key] if key is not None else ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    print("✅ 权重加载完毕")

    # 4. 取一个真实样本，测单张推理时间
    batch = next(iter(test_loader))
    images = batch["image"].to(device)      # [1, C, H, W]
    B, C, H, W = images.shape
    print(f"\n真实样本尺寸: batch={B}, C={C}, H={H}, W={W}")

    # 预热几次（让 cudnn / kernel 稳定）
    for _ in range(5):
        _ = model(images)

    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize()

    iters = 50  # 重复次数
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / iters
    fps = 1.0 / avg_time
    print(f"\n单张推理延迟 (真实尺寸): {avg_time*1000:.2f} ms  |  {fps:.1f} FPS")

    # 5. 用随机输入测大 batch 吞吐量（可选）
    batch_size = 8
    dummy = torch.randn(batch_size, C, H, W, device=device)
    for _ in range(5):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 30
    for _ in range(iters):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_imgs = iters * batch_size
    total_time = t1 - t0
    fps_batch = total_imgs / total_time
    print(f"\n批量推理吞吐量: batch={batch_size}, {fps_batch:.1f} FPS "
          f"({total_time/iters*1000:.2f} ms / batch)")


if __name__ == "__main__":
    benchmark()
