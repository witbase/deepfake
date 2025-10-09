import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from networks.vit_npr import create_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import concurrent.futures
from einops import rearrange, repeat

class AdvancedFeatureMapGenerator:
    def __init__(self, model_path, device='cuda'):
        """初始化特征图生成器"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # 创建模型
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
        
        # 用于保存各层特征的字典
        self.feature_maps = {}
        # 钩子句柄列表，用于移除钩子
        self.hook_handles = []
        
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
        
    def register_hooks(self):
        """注册钩子以获取不同层的特征图"""
        # 清空之前的钩子和特征图
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.feature_maps = {}
        
        # NPR特征提取器钩子
        def npr_hook(module, input, output):
            self.feature_maps['npr'] = output.detach()
        
        handle = self.model.npr_extract.register_forward_hook(npr_hook)
        self.hook_handles.append(handle)
        
        # NPR特征嵌入钩子
        for idx, embed in enumerate(self.model.npr_embeddings):
            def patch_embed_hook(idx):
                def hook(module, input, output):
                    self.feature_maps[f'patch_embed_{idx}'] = output.detach()
                return hook
            
            handle = embed.register_forward_hook(patch_embed_hook(idx))
            self.hook_handles.append(handle)
        
        # CLIP特征钩子
        def clip_hook(module, input, output):
            self.feature_maps['clip'] = output[-1].detach()  # 只获取最后一层特征
        
        # 注册fusion blocks钩子
        for idx, block in enumerate(self.model.fusion_blocks):
            def fusion_block_hook(idx):
                def hook(module, input, output):
                    self.feature_maps[f'fusion_{idx}'] = output.detach()
                return hook
            
            handle = block.register_forward_hook(fusion_block_hook(idx))
            self.hook_handles.append(handle)
            
        # 最终融合特征钩子
        def final_fusion_hook(module, input, output):
            self.feature_maps['final_fusion'] = output.detach()
            
        handle = self.model.norm.register_forward_hook(final_fusion_hook)
        self.hook_handles.append(handle)
    
    def generate_feature_maps(self, image_path, output_dir):
        """生成所有层的特征图"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 注册钩子
        self.register_hooks()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 保存原始图像
        image_name = os.path.basename(image_path)
        raw_image_path = os.path.join(output_dir, f"raw_{image_name}")
        image.save(raw_image_path)
        
        # 前向传播以获取特征图
        with torch.no_grad():
            self.model(image_tensor)
        
        # 处理和保存所有特征图
        for layer_name, feature_map in self.feature_maps.items():
            # NPR特征图处理
            if layer_name == 'npr':
                # 保存NPR特征图
                npr_map = feature_map.squeeze().cpu().permute(1, 2, 0).numpy()
                # 计算每个通道的平均值
                npr_map_mean = np.mean(npr_map, axis=2)
                # 归一化
                npr_map_norm = self.normalize_map(npr_map_mean)
                # 保存特征图
                save_path = os.path.join(output_dir, f"{layer_name}_{image_name}")
                cv2.imwrite(save_path, npr_map_norm)
                
                # 额外处理：边缘增强
                sobel_x = cv2.Sobel(npr_map_mean, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(npr_map_mean, cv2.CV_64F, 0, 1, ksize=3)
                edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
                edge_map = self.normalize_map(edge_map)
                # 保存边缘特征图
                edge_save_path = os.path.join(output_dir, f"{layer_name}_edge_{image_name}")
                cv2.imwrite(edge_save_path, edge_map)
                
            # Patch嵌入特征图处理
            elif layer_name.startswith('patch_embed'):
                # 处理patch embedding特征 (B, N, C) -> (N, C)
                patch_feats = feature_map.squeeze().cpu().numpy()
                # 取前1000个通道进行可视化（如果太多）
                vis_channels = min(1000, patch_feats.shape[1])
                reduced_feats = np.mean(patch_feats[:, :vis_channels], axis=1)
                
                # 重塑为图像格式
                patch_size = 16 if '0' in layer_name else 32  # 根据您的模型
                h = w = 256 // patch_size  # 假设输入是256x256
                try:
                    feat_map = reduced_feats.reshape(h, w)
                    # 归一化并保存
                    feat_map_norm = self.normalize_map(feat_map)
                    save_path = os.path.join(output_dir, f"{layer_name}_{image_name}")
                    cv2.imwrite(save_path, feat_map_norm)
                except:
                    print(f"无法重塑 {layer_name} 特征为图像，大小：{reduced_feats.shape}")
            
            # Fusion块特征图处理
            elif layer_name.startswith('fusion'):
                # 这些是注意力块的输出，只保留CLS token后的特征
                fusion_feats = feature_map.squeeze().cpu().numpy()
                if fusion_feats.shape[0] > 1:  # 确保有token
                    # 取CLS token后的特征，不包括CLS token
                    token_feats = fusion_feats[1:, :]  
                    # 平均所有通道
                    token_mean = np.mean(token_feats, axis=1)
                    
                    # 重塑为图像格式
                    total_patches = token_mean.shape[0]
                    h = w = int(np.sqrt(total_patches))
                    if h*w == total_patches:  # 确保是完美平方
                        feat_map = token_mean.reshape(h, w)
                        # 归一化并保存
                        feat_map_norm = self.normalize_map(feat_map)
                        save_path = os.path.join(output_dir, f"{layer_name}_{image_name}")
                        cv2.imwrite(save_path, feat_map_norm)
                    else:
                        print(f"无法重塑 {layer_name} 特征为图像，大小：{token_mean.shape}")

            # 最终融合特征处理
            elif layer_name == 'final_fusion':
                # 处理最终融合特征，只取CLS token后的特征
                final_feats = feature_map.squeeze().cpu().numpy()
                if final_feats.shape[0] > 1:  # 确保有token
                    token_feats = final_feats[1:, :]
                    token_mean = np.mean(token_feats, axis=1)
                    
                    # 重塑为图像格式
                    total_patches = token_mean.shape[0]
                    h = w = int(np.sqrt(total_patches))
                    if h*w == total_patches:
                        feat_map = token_mean.reshape(h, w)
                        # 归一化并保存
                        feat_map_norm = self.normalize_map(feat_map)
                        save_path = os.path.join(output_dir, f"{layer_name}_{image_name}")
                        cv2.imwrite(save_path, feat_map_norm)
        
        # 移除钩子
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        print(f"Feature maps generated for {image_path} in {output_dir}")
        return os.path.join(output_dir, f"npr_{image_name}")
        
    def normalize_map(self, feature_map):
        """归一化特征图到0-255范围"""
        feature_min = feature_map.min()
        feature_max = feature_map.max()
        if feature_max > feature_min:
            normalized = (feature_map - feature_min) / (feature_max - feature_min) * 255
        else:
            normalized = np.zeros_like(feature_map)
        return normalized.astype(np.uint8)
    
    def get_token_features(self, token_features):
        """处理各种形状的token特征，包括非完美平方数token
        
        Args:
            token_features: 形状为 [N, C] 的token特征矩阵
            
        Returns:
            feature_map: 重塑后的二维特征图
            success: 是否成功处理
        """
        # 移除CLS token，只保留patch tokens
        if token_features.shape[0] <= 1:
            print("特征中没有足够的tokens")
            return None, False
            
        patch_tokens = token_features[1:, :]  # 移除CLS token
        # 计算token特征的平均值
        token_mean = np.mean(patch_tokens, axis=1)
        
        # 处理方式1：尝试完美平方形重塑
        total_tokens = token_mean.shape[0]
        size = int(np.sqrt(total_tokens))
        
        if size * size == total_tokens:
            # 是完美平方数
            token_map = token_mean.reshape(size, size)
            return token_map, True
            
        # 处理方式2：自动寻找最优矩形布局
        factors = []
        for i in range(1, int(np.sqrt(total_tokens)) + 1):
            if total_tokens % i == 0:
                factors.append(i)
        
        if factors:
            # 找到最接近正方形的因子
            best_factor = factors[-1]
            height = best_factor
            width = total_tokens // best_factor
            
            token_map = token_mean.reshape(height, width)
            return token_map, True
            
        # 处理方式3：完成填充到最近的平方数
        next_square = int(np.ceil(np.sqrt(total_tokens))) ** 2
        padding = np.zeros(next_square - total_tokens)
        padded_tokens = np.concatenate([token_mean, padding])
        
        size = int(np.sqrt(next_square))
        token_map = padded_tokens.reshape(size, size)
        
        print(f"非完美平方数的token: {total_tokens}，填充到 {size}x{size}")
        return token_map, True
        
    def process_dataset(self, dataset_path, output_dir, max_images=100):
        """处理整个数据集中的图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for file in os.listdir(dataset_path):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_files.append(os.path.join(dataset_path, file))
        
        # 限制处理的图像数量
        if max_images > 0 and len(image_files) > max_images:
            print(f"限制处理前 {max_images} 张图像")
            image_files = image_files[:max_images]
        
        # 串行处理每个图像
        for img_path in tqdm(image_files, desc="处理图像"):
            img_name = os.path.basename(img_path)
            img_output_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
            self.generate_feature_maps(img_path, img_output_dir)
    
    def generate_composite_map(self, image_path, output_path):
        """生成复合特征图"""
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 注册钩子
        self.register_hooks()
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 前向传播以获取特征图
        with torch.no_grad():
            self.model(image_tensor)
        
        # 获取NPR特征和最终融合特征
        npr_feat = self.feature_maps.get('npr')
        fusion_feat = self.feature_maps.get('final_fusion')
        
        if npr_feat is not None and fusion_feat is not None:
            # 处理NPR特征
            npr_map = npr_feat.squeeze().cpu().permute(1, 2, 0).numpy()
            npr_map_mean = np.mean(npr_map, axis=2)
            # 提取边缘
            sobel_x = cv2.Sobel(npr_map_mean, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(npr_map_mean, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_map = self.normalize_map(edge_map)
            
            # 处理融合特征
            final_feats = fusion_feat.squeeze().cpu().numpy()
            if final_feats.shape[0] > 1:
                # 使用新添加的token特征处理方法
                fusion_map, success = self.get_token_features(final_feats)
                
                if success and fusion_map is not None:
                    # 归一化特征图
                    fusion_map = self.normalize_map(fusion_map)
                    
                    # 调整大小以匹配
                    fusion_map = cv2.resize(fusion_map, (edge_map.shape[1], edge_map.shape[0]))
                    
                    # 增强特征图的清晰度和对比度
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    edge_map_enhanced = clahe.apply(edge_map)
                    fusion_map_enhanced = clahe.apply(fusion_map)
                    
                    # 应用图像锐化增强边缘特征
                    edge_sharpened = cv2.addWeighted(edge_map_enhanced, 1.5, 
                                                  cv2.GaussianBlur(edge_map_enhanced, (5, 5), 0), -0.5, 0)
                    fusion_sharpened = cv2.addWeighted(fusion_map_enhanced, 1.5, 
                                                     cv2.GaussianBlur(fusion_map_enhanced, (5, 5), 0), -0.5, 0)
                    
                    # 使用强调重要区域的自适应混合方法
                    # 计算对比度参数
                    edge_std = np.std(edge_sharpened)
                    fusion_std = np.std(fusion_sharpened)
                    alpha = edge_std / (edge_std + fusion_std + 1e-6)
                    beta = 1.0 - alpha
                    
                    # 混合两个特征图 - 动态权重平衡特征清晰度
                    composite = cv2.addWeighted(edge_sharpened, 0.65, fusion_sharpened, 0.35, 0)
                    
                    # 再次提高整体对比度
                    composite = clahe.apply(composite)
                    
                    # 使用最佳热力图着色方案 - TURBO比JET对视觉更友好
                    heatmap = cv2.applyColorMap(composite, cv2.COLORMAP_TURBO)
                    
                    # 增强热力图的饱和度
                    heatmap_hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
                    heatmap_hsv[:,:,1] = np.clip(heatmap_hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)  # 增加饱和度
                    heatmap = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2BGR)
                    
                    # 将原图像调整大小
                    orig_img = np.array(image.resize((edge_map.shape[1], edge_map.shape[0])))
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                    
                    # 将热力图叠加到原图上 - 更倾向于显示热力图
                    overlay = cv2.addWeighted(orig_img, 0.3, heatmap, 0.7, 0)
                    
                    # 保存结果
                    cv2.imwrite(output_path, overlay)
                    
                    # 额外保存纯热力图版本
                    heatmap_path = output_path.replace('composite_', 'heatmap_')
                    cv2.imwrite(heatmap_path, heatmap)
                    
                    # 移除钩子
                    for handle in self.hook_handles:
                        handle.remove()
                    self.hook_handles = []
                    
                    print(f"Composite feature map saved to {output_path}")
                    return output_path
        
        # 移除钩子
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        print("Failed to generate composite map")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Generate comprehensive feature maps')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to the input image or directory')
    parser.add_argument('--output_dir', type=str, default='./feature_maps',
                      help='Directory to save feature maps')
    parser.add_argument('--max_images', type=int, default=100,
                      help='Maximum number of images to process if input is a directory')
    parser.add_argument('--mode', type=str, default='composite',
                      choices=['all', 'composite'],
                      help='Mode of operation: all for all feature maps, composite for single overlay map')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建特征图生成器
    generator = AdvancedFeatureMapGenerator(args.model_path)
    
    # 确定输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个图像
        if args.mode == 'all':
            # 生成所有层的特征图
            output_dir = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input))[0])
            generator.generate_feature_maps(args.input, output_dir)
        else:
            # 生成复合特征图
            output_path = os.path.join(args.output_dir, 
                                     f"composite_{os.path.basename(args.input)}")
            generator.generate_composite_map(args.input, output_path)
    else:
        # 处理目录中的图像
        if args.mode == 'all':
            # 生成所有层的特征图
            generator.process_dataset(args.input, args.output_dir, args.max_images)
        else:
            # 为每个图像生成复合特征图
            output_dir = os.path.join(args.output_dir, 'composite_maps')
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有图像文件
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            image_files = []
            
            for file in os.listdir(args.input):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_files.append(os.path.join(args.input, file))
            
            # 限制处理的图像数量
            if args.max_images > 0 and len(image_files) > args.max_images:
                print(f"限制处理前 {args.max_images} 张图像")
                image_files = image_files[:args.max_images]
            
            for img_path in tqdm(image_files, desc="生成特征图"):
                output_path = os.path.join(output_dir, f"composite_{os.path.basename(img_path)}")
                generator.generate_composite_map(img_path, output_path)

if __name__ == "__main__":
    main() 