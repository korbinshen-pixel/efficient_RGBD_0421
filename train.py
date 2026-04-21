"""RGB-D 单流训练脚本
使用:
- RGB 图像
- depth_pc 投影得到的 depth map
融合成 4 通道输入 [B, 4, H, W]
"""

import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from models.efficientpose import EfficientPose
from datasets.pallet_dataset import PalletDataset, collate_fn
from losses.pose_loss import PoseLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_losses = {}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)   # [B,4,H,W]

        predictions = model(images)

        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ['image', 'image_id', 'image_ids']
        }

        losses = criterion(predictions, targets)

        # 第一个 batch 的诊断
        if batch_idx == 0:
            from losses.pose_loss import compute_iou

            pred_bbox = predictions['bbox']
            gt_bbox = targets['bbox']
            iou = compute_iou(pred_bbox, gt_bbox)
            pos_mask = (iou > 0.5).float()
            neg_mask = (iou < 0.4).float()

            print(f"\n{'='*60}")
            print(f"【Anchor 分配诊断】")
            print(f"  Batch size: {images.shape[0]}")
            print(f"  Input shape: {images.shape}")
            print(f"  Total anchors: {pred_bbox.shape[1]}")
            print(f"  正样本数: {pos_mask.sum().item():.0f} ({pos_mask.sum().item()/images.shape[0]:.1f}/图)")
            print(f"  负样本数: {neg_mask.sum().item():.0f} ({neg_mask.sum().item()/images.shape[0]:.1f}/图)")
            print(f"  最大 IoU: {iou.max().item():.4f}")
            print(f"  IoU > 0.5: {(iou > 0.5).sum().item()}")
            print(f"  IoU > 0.3: {(iou > 0.3).sum().item()}")
            print(f"  IoU > 0.1: {(iou > 0.1).sum().item()}")

            conf = predictions['class'].squeeze(-1)
            print(f"\n【置信度分布】")
            print(f"  均值: {conf.mean().item():.4f}  最大: {conf.max().item():.4f}  最小: {conf.min().item():.4f}")

            print(f"\n【RGB-D 输入检查】")
            print(f"  image shape: {images.shape}")
            if images.shape[1] >= 4:
                rgb_mean = images[:, :3].mean().item()
                depth_mean = images[:, 3:4].mean().item()
                depth_max = images[:, 3:4].max().item()
                depth_min = images[:, 3:4].min().item()
                print(f"  RGB mean:   {rgb_mean:.4f}")
                print(f"  Depth mean: {depth_mean:.4f}")
                print(f"  Depth min/max: {depth_min:.4f} / {depth_max:.4f}")
            print(f"{'='*60}\n")

        optimizer.zero_grad()

        

        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            print(f"⚠️ Epoch {epoch} Batch {batch_idx}: loss 异常，跳过")
            optimizer.zero_grad()
            continue

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key, val in losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + float(val.item())

        pbar.set_postfix({'loss': losses['total'].item()})

    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_losses = {}

    pbar = tqdm(dataloader, desc='         [Val]  ')
    for batch in pbar:
        images = batch['image'].to(device)   # [B,4,H,W]

        predictions = model(images)

        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k not in ['image', 'image_id', 'image_ids']
        }

        losses = criterion(predictions, targets)

        for key, val in losses.items():
            total_losses[key] = total_losses.get(key, 0.0) + float(val.item())

        pbar.set_postfix({'loss': losses['total'].item()})

    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    return avg_losses


def print_losses(prefix, losses):
    parts = [f'{prefix}']
    for key, val in losses.items():
        parts.append(f'{key}={val:.4f}')
    print('  '.join(parts))


def main():
    config = Config()
    device = torch.device(config.device)

    Path(config.checkpoints_dir).mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter('runs/experiment_rgbd')
    loss_history = {'train': [], 'val': []}

    train_dataset = PalletDataset(
        config.dataset_path,
        config.object_dir,
        'train',
        config=config
    )
    val_dataset = PalletDataset(
        config.dataset_path,
        config.object_dir,
        'test',
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory
    )

    # 单流 RGB-D：输入通道 = 4
    model = EfficientPose(
        phi=config.phi,
        num_classes=config.num_classes,
        pretrained=True,
        in_channels=4
    ).to(device)

    criterion = PoseLoss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    best_val_loss = float('inf')

    print("\n" + "=" * 80)
    print("开始 RGB-D 单流训练")
    print(f"设备: {device}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"输入通道数: 4 (RGB + Depth)")
    print("=" * 80 + "\n")

    for epoch in range(1, config.num_epochs + 1):
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print_losses(f'Epoch {epoch} Train |', train_losses)

        val_losses = validate(
            model, val_loader, criterion, device
        )
        print_losses(f'Epoch {epoch} Val   |', val_losses)

        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)

        for key in train_losses:
            writer.add_scalar(f'Loss_detail/{key}_train', train_losses[key], epoch)
            writer.add_scalar(f'Loss_detail/{key}_val', val_losses.get(key, 0.0), epoch)

        loss_history['train'].append(train_losses)
        loss_history['val'].append(val_losses)

        with open('loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=2)

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = f'{config.checkpoints_dir}/best_model_phi{config.phi}_rgbd.pth'
            torch.save(model.state_dict(), save_path)
            print(f'✅ 保存最佳模型: {save_path} (val_loss={best_val_loss:.4f})')

        # 每轮保存最新模型
        latest_path = f'{config.checkpoints_dir}/latest_model_phi{config.phi}_rgbd.pth'
        torch.save(model.state_dict(), latest_path)

        print('-' * 80)

    writer.close()
    print("训练结束。")


if __name__ == '__main__':
    main()