import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from networks.vit_npr import create_model
from tqdm import tqdm
import argparse
import concurrent.futures

class FeatureMapGenerator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model(num_classes=1)
        self.load_model(model_path)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """加载预训练模型"""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        print("Model loaded successfully")

    def get_feature_layer(self):
        """获取特征层"""
        if hasattr(self.model, 'npr_embeddings') and len(self.model.npr_embeddings) > 0:
            return self.model.npr_embeddings[0]
        else:
            raise ValueError('模型不包含可用的特征层')

    def generate_feature_map(self, image_path, save_path):
        """生成特征图"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 选择特征层
        feature_layer = self.get_feature_layer()
        
        # 注册钩子以获取特征图
        def forward_hook(module, input, output):
            self.feature_map = output.detach()
        
        hook_handle = feature_layer.register_forward_hook(forward_hook)
        
        # 前向传播以获取特征图
        with torch.no_grad():
            self.model(image_tensor)
        
        # 移除钩子
        hook_handle.remove()
        
        # 处理特征图（例如，边缘增强）
        feature_map = self.feature_map.squeeze().cpu().numpy()
        feature_map = np.mean(feature_map, axis=0)  # 平均通道
        
        # 应用边缘检测（如Sobel算子）
        sobel_x = cv2.Sobel(feature_map, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(feature_map, cv2.CV_64F, 0, 1, ksize=5)
        edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
        
        # 保存特征图
        cv2.imwrite(save_path, (edge_map * 255).astype(np.uint8))
        print(f"Feature map saved to {save_path}")

    def process_single_image(self, img_path, save_path):
        """处理单个图像并生成特征图"""
        try:
            self.generate_feature_map(img_path, save_path)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    def process_dataset(self, dataset_path, output_dir):
        """处理整个数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像
        images = os.listdir(dataset_path)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for img_name in images:
                img_path = os.path.join(dataset_path, img_name)
                save_path = os.path.join(output_dir, f"feature_map_{img_name}")
                futures.append(executor.submit(self.process_single_image, img_path, save_path))
            
            # 等待所有线程完成
            for future in concurrent.futures.as_completed(futures):
                future.result()  # 获取结果以处理异常

def parse_args():
    parser = argparse.ArgumentParser(description='Generate feature maps for images')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./feature_maps',
                      help='Directory to save feature maps')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建特征图生成器
    generator = FeatureMapGenerator(args.model_path, device='cuda')
    
    # 处理数据集
    generator.process_dataset(args.dataset_path, args.output_dir)

if __name__ == "__main__":
    main()