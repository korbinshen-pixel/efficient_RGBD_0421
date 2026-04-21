"""推理脚本"""
import torch
import cv2
import numpy as np
from config import Config
from models.efficientpose import EfficientPose

def inference(image_path, model_path, config):
    device = torch.device(config.device)

    # 加载模型
    model = EfficientPose(phi=config.phi, num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)

    # 提取结果（取第一个检测）
    bbox = outputs['bbox'][0, 0].cpu().numpy()
    rotation = outputs['rotation'][0, 0].cpu().numpy().reshape(3, 3)
    translation = outputs['translation'][0, 0].cpu().numpy()

    print(f"检测到托盘:")
    print(f"  2D Bbox: {bbox}")
    print(f"  旋转矩阵:\n{rotation}")
    print(f"  平移向量: {translation}")

    return bbox, rotation, translation

if __name__ == '__main__':
    config = Config()
    inference('test.png', f'{config.checkpoints_dir}/best_model_phi0.pth', config)
