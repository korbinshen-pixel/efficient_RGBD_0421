"""
Pallet Dataset for EfficientPose Training
支持 RGB + Depth PointCloud 双流输入（融合为 4 通道 RGB-D）
当前版本:
- 不再使用 LiDAR (lidar/*.npy)
- 使用 depth_pc/*.npy 中的点云，投影成 depth map
- 最终返回 image: [4, H, W] (RGB 正则化 + 深度归一化)
"""

import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from config import Config


def pointcloud_to_depthmap(points,
                           fx: float = None,
                           fy: float = None,
                           cx: float = None,
                           cy: float = None,
                           h: int = 480,
                           w: int = 640,
                           z_min: float = 0.1,
                           z_max: float = 15.0) -> np.ndarray:
    """
    点云 -> 深度图

    支持两种输入格式：
    1) (H, W, 3)  像素对齐 XYZ 图 -> 直接取 Z 通道
    2) (N, 3)     稀疏点云        -> 用内参投影到图像平面
    """
    if points is None:
        return np.zeros((h, w), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    # ── 格式 1: (H, W, 3) 像素对齐 XYZ 图 ──
    if points.ndim == 3 and points.shape[2] == 3:
        depth_map = points[:, :, 2].copy()          # 取 Z 通道
        depth_map[(depth_map < z_min) | (depth_map > z_max)] = 0.0
        depth_map[~np.isfinite(depth_map)] = 0.0
        return depth_map.astype(np.float32)

    # ── 格式 2: (N, 3) 稀疏点云 ──
    depth_map = np.zeros((h, w), dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return depth_map

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) \
            & (z > z_min) & (z < z_max)
    if not np.any(valid):
        return depth_map

    x, y, z = x[valid], y[valid], z[valid]

    u = np.round(fx * x / z + cx).astype(np.int32)
    v = np.round(fy * y / z + cy).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]

    for ui, vi, zi in zip(u, v, z):
        d_old = depth_map[vi, ui]
        if d_old == 0.0 or zi < d_old:
            depth_map[vi, ui] = zi

    return depth_map


class PalletDataset(Dataset):
    """托盘数据集 - Linemod 格式，支持 RGB + Depth PointCloud 输入"""

    def __init__(self, dataset_path, object_dir='01', split='train',
                 transform=None, config=None):
        self.dataset_path = Path(dataset_path).expanduser()
        self.data_dir = self.dataset_path / object_dir
        self.split = split
        self.transform = transform
        self.config = config

        print(f"正在加载数据集: {self.data_dir}")

        split_file = self.data_dir / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"找不到 {split_file}")

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        with open(self.data_dir / 'gt.yml', 'r') as f:
            self.gt_dict = yaml.safe_load(f)
        with open(self.data_dir / 'info.yml', 'r') as f:
            self.info_dict = yaml.safe_load(f)

        # 深度点云目录 (由深度图转点云后的 Nx3 保存)
        self.depth_pc_dir = self.data_dir / 'depth_pc'
        self.use_depth_pc = self.depth_pc_dir.exists()
        self.use_lidar = False   # 显式关闭 LiDAR
        self.use_depth = False   # 不再单独使用原始深度图

        if self.use_depth_pc:
            print(f"✅ 检测到 depth_pc 目录: {self.depth_pc_dir}")
            # 初始化时检查第一个样本是否真的存在对应 .npy
            first_id = self.image_ids[0]
            first_pc = self.depth_pc_dir / f'{first_id}.npy'
            if first_pc.exists():
                probe = np.load(str(first_pc))
                print(f"   首帧 depth_pc shape={probe.shape}, dtype={probe.dtype}")
            else:
                print(f"⚠️  注意: 首帧 {first_pc} 不存在，请确认文件命名格式！")
        else:
            print(f"⚠️ 未找到 depth_pc 目录 ({self.depth_pc_dir})，深度通道将全为 0")

        print(f"✅ 加载 {split} 数据集: {len(self.image_ids)} 帧")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        frame_id = int(image_id)

        # ---------- RGB ----------
        rgb_path = self.data_dir / 'rgb' / f'{image_id}.png'
        image = cv2.imread(str(rgb_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # ---------- 相机内参 ----------
        if frame_id not in self.info_dict:
            raise KeyError(f"info.yml 中找不到帧 {frame_id}")

        info_data = self.info_dict[frame_id]
        info = info_data[0] if isinstance(info_data, list) else info_data
        K = np.array(info['cam_K'], dtype=np.float32).reshape(3, 3)

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # ---------- mask ----------
        mask_path = self.data_dir / 'mask' / f'{image_id}.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.ones((orig_h, orig_w), dtype=np.uint8)

        # ---------- bbox（先用原图 mask 算） ----------
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        else:
            bbox = np.array([0, 0, orig_w - 1, orig_h - 1], dtype=np.float32)

        bbox = bbox / np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)

        # ---------- resize ----------
        target_size = self.config.image_size if self.config is not None else Config.image_size
        image = cv2.resize(image, (target_size, target_size))
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # ---------- Depth PointCloud -> depth map ----------
        depth_map = np.zeros((target_size, target_size), dtype=np.float32)
        depth_full_res = np.zeros((orig_h, orig_w), dtype=np.float32)  # debug 用，提前初始化

        if self.use_depth_pc:
            depth_pc_path = self.depth_pc_dir / f'{image_id}.npy'

            if depth_pc_path.exists():
                points = np.load(str(depth_pc_path)).astype(np.float32)

                depth_full_res = pointcloud_to_depthmap(
                    points,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    h=orig_h, w=orig_w,
                    z_min=0.1, z_max=15.0
                )

                depth_map = cv2.resize(
                    depth_full_res,
                    (target_size, target_size),
                    interpolation=cv2.INTER_NEAREST
                )

                # 简单补洞：用 3x3 高斯平滑填补 0 区域
                depth_blur = cv2.GaussianBlur(depth_map, (3, 3), 0)
                zero_mask = (depth_map == 0)
                depth_map[zero_mask] = depth_blur[zero_mask]

            else:
                if idx < 3:
                    print(f"⚠️  depth_pc 文件不存在: {depth_pc_path}, 此帧深度置为 0")

            # # ★ DEBUG：打印前 4 个样本的深度诊断信息
            # if idx < 4:
            #     print(f"\n[DEBUG depth] idx={idx}, image_id={image_id}")
            #     print(f"  depth_pc_dir : {self.depth_pc_dir}")
            #     print(f"  depth_pc_path: {depth_pc_path}")
            #     print(f"  path exists  : {depth_pc_path.exists()}")
            #     if depth_pc_path.exists():
            #         pts_dbg = np.load(str(depth_pc_path))
            #         print(f"  points shape : {pts_dbg.shape}, dtype={pts_dbg.dtype}")
            #         if pts_dbg.ndim == 2 and pts_dbg.shape[1] >= 3:
            #             z_col = pts_dbg[:, 2]
            #             print(f"  z min/max    : {float(z_col.min()):.3f} / {float(z_col.max()):.3f}")
            #             print(f"  valid z (0.1~15m): {int(((z_col > 0.1) & (z_col < 15.0)).sum())} / {len(z_col)}")
            #         else:
            #             print(f"  ⚠️ points 形状异常，不是 Nx3！")
            #         print(f"  depth_full_res nonzero: {int((depth_full_res > 0).sum())}, "
            #               f"max={float(depth_full_res.max()):.3f}")
            #         print(f"  depth_map nonzero     : {int((depth_map > 0).sum())}, "
            #               f"max={float(depth_map.max()):.3f}")
            #     print()

        # ---------- pose label ----------
        if frame_id not in self.gt_dict:
            raise KeyError(f"gt.yml 中找不到帧 {frame_id}")

        gt_data = self.gt_dict[frame_id]
        gt = gt_data[0] if isinstance(gt_data, list) else gt_data

        R = np.array(gt['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        t = np.array(gt['cam_t_m2c'], dtype=np.float32) / 1000.0  # mm -> m

        # ---------- augment ----------
        # 注意: 如果 transform 会改变几何形状（如旋转、缩放），需要同时对 depth_map 做相同处理
        # 这里假定 transform 只做颜色增强（如 ColorJitter），不会改变几何
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # ---------- RGB + 深度 归一化 & 拼接 ----------
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [3,H,W]

        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        image_t = (image_t - mean) / std

        # 深度归一化到 0~1（用 z_max 15m）
        depth_t = torch.from_numpy(depth_map.astype(np.float32))
        valid_d = depth_t > 0
        depth_t = depth_t / 15.0
        depth_t[~valid_d] = 0.0
        depth_t = depth_t.unsqueeze(0)  # [1,H,W]

        # 拼成 4 通道 RGB-D
        image_4ch = torch.cat([image_t, depth_t], dim=0)  # [4,H,W]

        return {
            'image': image_4ch,                   # [4, H, W]
            'bbox': torch.from_numpy(bbox).float(),
            'rotation': torch.from_numpy(R).float(),
            'translation': torch.from_numpy(t).float(),
            'camera_matrix': torch.from_numpy(K).float(),
            'mask': torch.from_numpy(mask.astype(np.float32)),
            'image_id': image_id
        }


def collate_fn(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'bbox': torch.stack([b['bbox'] for b in batch]),
        'rotation': torch.stack([b['rotation'] for b in batch]),
        'translation': torch.stack([b['translation'] for b in batch]),
        'camera_matrix': torch.stack([b['camera_matrix'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'image_ids': [b['image_id'] for b in batch]
    }