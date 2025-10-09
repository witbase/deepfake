import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from networks.vit_npr import create_model
import cv2
from tqdm import tqdm
import argparse

class SwinGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[:, class_idx]
        loss.backward()
        
        # 获取梯度和激活
        gradients = self.gradients
        activations = self.activations
        
        # 改进权重计算方式，使用绝对值平均以捕获负贡献
        if len(gradients.shape) == 3:  # [B, N, C]
            weights = torch.mean(torch.abs(gradients), dim=1, keepdim=True)  # [B, 1, C]
        else:
            weights = torch.mean(torch.abs(gradients), dim=(1, 2), keepdim=True)  # [B, 1, 1, C]
        
        # 生成CAM
        if len(activations.shape) == 3:  # [B, N, C]
            # 计算每个patch的权重
            cam = (weights * activations).sum(dim=-1)  # [B, N]
            
            # 获取输入图像的高度和宽度
            h, w = input_tensor.shape[2:]
            
            # 计算patch数量
            num_patches = cam.shape[1]
            
            # 打印调试信息
            print(f"Input shape: {input_tensor.shape}")
            print(f"Number of patches: {num_patches}")
            print(f"Activation shape: {activations.shape}")
            
            # 计算特征图的网格大小
            # 使用固定的patch size (16x16)
            patch_size = 16
            grid_h = h // patch_size
            grid_w = w // patch_size
            
            # 计算实际需要的patch数量
            expected_patches = grid_h * grid_w
            
            # 如果patch数量不匹配，进行填充或截断
            if num_patches > expected_patches:
                # 截断多余的patch
                cam = cam[:, :expected_patches]
            elif num_patches < expected_patches:
                # 填充缺失的patch
                padding = torch.zeros((cam.shape[0], expected_patches - num_patches), device=cam.device)
                cam = torch.cat([cam, padding], dim=1)
            
            print(f"Grid size: {grid_h}x{grid_w}")
            
            # 重塑为2D特征图
            cam = cam.reshape(cam.shape[0], grid_h, grid_w)
            
            # 上采样到原始图像大小，使用bicubic插值获得更平滑的结果
            cam = F.interpolate(
                cam.unsqueeze(1),  # [B, 1, H, W]
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
        else:
            cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 应用ReLU后进行归一化，但保留更多的中间值以获得更丰富的热力图
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        
        # 应用高斯模糊使热力图更平滑
        cam = cv2.GaussianBlur(cam, (5, 5), 1.0)
        
        # 使用CLAHE增强对比度
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam = (cam * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cam = clahe.apply(cam)
            cam = cam.astype(float) / 255.0
        else:
            cam = np.zeros_like(cam)
        
        return cam

class MultiLayerGradCAM:
    """使用多层特征生成更全面的热力图"""
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.gradients = {}
        self.activations = {}
        
    def _register_hooks(self, target_layers):
        """注册多层钩子"""
        for name, layer in target_layers.items():
            def forward_hook(name):
                def hook(module, input, output):
                    self.activations[name] = output.detach()
                return hook
                
            def backward_hook(name):
                def hook(module, grad_in, grad_out):
                    self.gradients[name] = grad_out[0].detach()
                return hook
            
            self.hooks.append(layer.register_forward_hook(forward_hook(name)))
            self.hooks.append(layer.register_backward_hook(backward_hook(name)))
            
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def __call__(self, input_tensor, target_layers, class_idx=None):
        """
        生成多层融合的Grad-CAM热力图
        Args:
            input_tensor: 输入张量
            target_layers: 目标层的字典 {layer_name: layer}
            class_idx: 类别索引
        Returns:
            融合的热力图
        """
        h, w = input_tensor.shape[2:]
        # 注册钩子
        self._register_hooks(target_layers)
        
        # 前向传播
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # 如果未指定类别，使用最高概率的类别
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        loss = output[:, class_idx]
        loss.backward()
        
        # 处理所有层的热力图
        layer_cams = {}
        for name, layer in target_layers.items():
            if name not in self.gradients or name not in self.activations:
                print(f"Layer {name} not found in activations or gradients")
                continue
                
            gradients = self.gradients[name]
            activations = self.activations[name]
            
            # 处理不同类型的特征
            if len(activations.shape) == 3:  # [B, N, C] - tokens
                # 使用绝对值梯度的平均
                weights = torch.mean(torch.abs(gradients), dim=1, keepdim=True)
                cam = (weights * activations).sum(dim=-1)  # [B, N]
                
                # 重塑为2D特征图
                # 计算网格大小
                patch_size = 16  # 假设patch大小为16x16
                grid_h = h // patch_size
                grid_w = w // patch_size
                
                # 计算应有的patch数量
                expected_patches = grid_h * grid_w
                
                # 适应不同的token数量
                if cam.shape[1] > 1:  # 至少有CLS token后的token
                    # 移除CLS token (如果有)
                    cam = cam[:, 1:]
                    
                    if cam.shape[1] != expected_patches:
                        if cam.shape[1] > expected_patches:
                            # 截断多余的token
                            cam = cam[:, :expected_patches]
                        else:
                            # 填充缺失的token
                            padding = torch.zeros((cam.shape[0], expected_patches - cam.shape[1]), 
                                               device=cam.device)
                            cam = torch.cat([cam, padding], dim=1)
                    
                    # 重塑为2D并上采样
                    cam = cam.reshape(cam.shape[0], grid_h, grid_w)
                else:
                    # 如果没有足够的token，创建空的cam
                    cam = torch.zeros((1, grid_h, grid_w), device=input_tensor.device)
            else:
                # 卷积特征 [B, C, H', W']
                weights = torch.mean(torch.abs(gradients), dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
            
            # 上采样到原始尺寸
            cam = F.interpolate(
                cam.unsqueeze(1) if len(cam.shape) == 3 else cam, 
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
            
            # 应用ReLU
            cam = F.relu(cam)
            
            # 归一化
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)
            
            layer_cams[name] = cam
        
        # 移除钩子
        self.remove_hooks()
        
        # 如果没有有效的cam，返回空图
        if not layer_cams:
            return np.zeros((h, w))
        
        # 融合所有层的热力图
        # 方法1: 平均融合
        # combined_cam = np.mean(list(layer_cams.values()), axis=0)
        
        # 方法2: 加权融合 - 给予更深层更高的权重
        weights = {}
        total_weight = 0
        for i, name in enumerate(layer_cams.keys()):
            weight = i + 1  # 越深的层权重越大
            weights[name] = weight
            total_weight += weight
            
        combined_cam = np.zeros_like(next(iter(layer_cams.values())))
        for name, cam in layer_cams.items():
            combined_cam += (weights[name] / total_weight) * cam
        
        # 应用高斯模糊使热力图更平滑
        combined_cam = cv2.GaussianBlur(combined_cam, (5, 5), 1.0)
        
        # 使用CLAHE增强对比度
        combined_cam = (combined_cam * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        combined_cam = clahe.apply(combined_cam)
        combined_cam = combined_cam.astype(float) / 255.0
        
        return combined_cam

class HeatmapGenerator:
    def __init__(self, model_path, device='cuda', feature_type='fusion'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model(num_classes=1)
        self.load_model(model_path)
        self.model.eval()
        self.feature_type = feature_type
        
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

    def get_target_layer(self):
        """获取目标层"""
        if self.feature_type == 'npr':
            # 只用高频特征，hook npr_embeddings[0]
            if hasattr(self.model, 'npr_embeddings') and len(self.model.npr_embeddings) > 0:
                return self.model.npr_embeddings[0]
            else:
                raise ValueError('模型不包含可用的npr特征层')
        elif self.feature_type == 'clip':
            if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'conv1'):
                print("使用CLIP特征层:", self.model.visual.conv1)
                return self.model.visual.conv1
            else:
                raise ValueError('模型不包含clip特征层')
        else:
            return self.model.fusion_blocks[len(self.model.fusion_blocks)//2]
    
    def get_multi_target_layers(self):
        """获取用于多层融合的目标层"""
        layers = {}
        
        # 尝试添加NPR层
        if hasattr(self.model, 'npr_extract'):
            layers['npr'] = self.model.npr_extract
            
        # 尝试添加NPR嵌入层
        if hasattr(self.model, 'npr_embeddings') and len(self.model.npr_embeddings) > 0:
            layers['npr_embed'] = self.model.npr_embeddings[0]
            
        # 尝试添加CLIP层
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'conv1'):
            layers['clip'] = self.model.visual.conv1
            
        # 添加融合层
        if hasattr(self.model, 'fusion_blocks'):
            # 添加第一层、中间层和最后一层
            num_blocks = len(self.model.fusion_blocks)
            if num_blocks >= 1:
                layers['fusion_first'] = self.model.fusion_blocks[0]
            if num_blocks >= 2:
                middle_idx = num_blocks // 2
                layers['fusion_middle'] = self.model.fusion_blocks[middle_idx]
            if num_blocks >= 3:
                layers['fusion_last'] = self.model.fusion_blocks[-1]
                
        # 添加最终normalization层
        if hasattr(self.model, 'norm'):
            layers['final'] = self.model.norm
        
        return layers

    def process_window_attention(self, attn_weights, window_size=7):
        """处理Swin Transformer的窗口注意力权重"""
        B, H, N, N = attn_weights.shape
        h = w = int(np.sqrt(N))
        
        # 重塑注意力权重
        attn_map = attn_weights.mean(dim=1)  # 平均所有头
        attn_map = attn_map.reshape(B, h, w, h, w)
        
        # 处理窗口注意力
        window_attn = torch.zeros(B, h, w)
        for i in range(h):
            for j in range(w):
                window_attn[:, i, j] = attn_map[:, i, j, i, j]
        
        return window_attn
        
    def generate_heatmap(self, image_path, save_path, overlay_alpha=0.5, colormap=cv2.COLORMAP_JET, 
                        use_multi_layer=True):
        """生成热力图
        
        Args:
            image_path: 输入图像路径
            save_path: 保存路径
            overlay_alpha: 热力图叠加透明度 (0-1)，越大越透明
            colormap: 热力图颜色映射方案
            use_multi_layer: 是否使用多层融合
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        try:
            if use_multi_layer:
                # 使用多层融合方法
                multi_layer_cam = MultiLayerGradCAM(self.model)
                target_layers = self.get_multi_target_layers()
                
                print(f"使用多层融合热力图，包含层: {list(target_layers.keys())}")
                cam = multi_layer_cam(image_tensor, target_layers)
            else:
                # 使用单层方法
                target_layer = self.get_target_layer()
                swin_gradcam = SwinGradCAM(self.model, target_layer)
                cam = swin_gradcam(image_tensor)
                swin_gradcam.remove_hooks()
            
            # 调整回原始图像大小
            cam = cv2.resize(cam, original_size)
            
            # 增强对比度
            # 应用CLAHE增强热力图对比度
            cam = (cam * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cam = clahe.apply(cam)
            
            # 锐化处理
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            cam = cv2.filter2D(cam, -1, kernel)
            
            # 归一化
            cam = cam.astype(float) / 255.0
            
            # 应用颜色映射前增强亮区和暗区的对比
            # 非线性增强，使突出区域更突出，低响应区域更抑制
            cam = np.power(cam, 0.7)  # gamma校正，增强中高值区域
            
            # 创建热力图
            heatmap = np.uint8(255 * cam)
            
            # 保存纯热力图 - 按参考图样式生成清晰的伪彩色图
            pure_heatmap_path = save_path.replace('heatmap_', 'pure_heatmap_')
            if use_multi_layer:
                pure_heatmap_path = pure_heatmap_path.replace('pure_heatmap_', 'pure_heatmap_multi_')
            
            # 创建不同着色方案的热力图
            colored_heatmap_turbo = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
            colored_heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            colored_heatmap_inferno = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
            
            # 保存不同版本的纯热力图
            cv2.imwrite(pure_heatmap_path.replace('.jpg', '_turbo.jpg').replace('.png', '_turbo.png'), 
                      colored_heatmap_turbo)
            cv2.imwrite(pure_heatmap_path.replace('.jpg', '_jet.jpg').replace('.png', '_jet.png'), 
                      colored_heatmap_jet)
            cv2.imwrite(pure_heatmap_path.replace('.jpg', '_inferno.jpg').replace('.png', '_inferno.png'), 
                      colored_heatmap_inferno)
            
            # 将原始图像转换为numpy数组
            original_image = np.array(image)
            
            # 使用参考图中类似的颜色映射方案
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
            
            # 增强热力图颜色的饱和度
            colored_heatmap_hsv = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2HSV)
            colored_heatmap_hsv[:,:,1] = np.clip(colored_heatmap_hsv[:,:,1] * 1.4, 0, 255).astype(np.uint8)
            colored_heatmap = cv2.cvtColor(colored_heatmap_hsv, cv2.COLOR_HSV2BGR)
            
            # 创建参考图类似的叠加效果
            # 创建透明热力图遮罩
            heat_mask = np.zeros_like(original_image, dtype=np.float32)
            heat_mask[..., :] = colored_heatmap / 255.0
            
            # 基于热力图强度计算alpha (自适应透明度) - 调整为更接近参考图
            alpha_mask = np.clip(cam * 2.5, 0.3, 0.85)  # 增加透明度范围，使热点更突出
            alpha_mask = np.expand_dims(alpha_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2)
            
            # 混合热力图和原始图像 - 生成类似参考图的效果
            overlay = (1.0 - alpha_mask) * (original_image / 255.0) + alpha_mask * heat_mask
            overlay = (overlay * 255).astype(np.uint8)
            
            # 保存叠加结果
            suffix = "multi_" if use_multi_layer else ""
            final_save_path = save_path.replace("heatmap_", f"heatmap_{suffix}")
            cv2.imwrite(final_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # 创建纯热力图（不叠加）
            cv2.imwrite(pure_heatmap_path, colored_heatmap)
            
            # 生成参考图风格的纯热力图（无背景）- 这个更接近您提供的参考图
            # 创建带透明通道的热力图
            heat_alpha = np.ones((heatmap.shape[0], heatmap.shape[1]), dtype=np.uint8) * 255
            heat_alpha[heatmap < 30] = 0  # 低响应区域设为透明
            
            # 合并RGB和Alpha通道
            colored_alpha = cv2.merge([colored_heatmap[:,:,0], colored_heatmap[:,:,1], 
                                     colored_heatmap[:,:,2], heat_alpha])
            
            # 保存带透明通道的PNG
            transparent_path = pure_heatmap_path.replace('.jpg', '_transparent.png').replace('.png', '_transparent.png')
            cv2.imwrite(transparent_path, colored_alpha)
            
            # 为了更好地展示，将热力图应用于纯白背景
            white_bg = np.ones_like(original_image) * 255
            white_overlay = (1.0 - alpha_mask) * (white_bg / 255.0) + alpha_mask * heat_mask
            white_overlay = (white_overlay * 255).astype(np.uint8)
            
            # 保存白底热力图
            white_path = pure_heatmap_path.replace('.jpg', '_white_bg.jpg').replace('.png', '_white_bg.png')
            cv2.imwrite(white_path, cv2.cvtColor(white_overlay, cv2.COLOR_RGB2BGR))
            
            return overlay
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_dataset(self, dataset_path, output_dir, num_samples=None, colormap=cv2.COLORMAP_JET):
        """处理整个数据集"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查数据集结构
        if os.path.isdir(dataset_path):
            # 获取所有伪造类型
            items = os.listdir(dataset_path)
            
            if any(os.path.isdir(os.path.join(dataset_path, item)) for item in items):
                # 分类文件夹结构
                fake_types = [item for item in items if os.path.isdir(os.path.join(dataset_path, item))]
                
                for fake_type in fake_types:
                    # 为每种伪造类型创建子目录
                    type_dir = os.path.join(output_dir, fake_type)
                    os.makedirs(type_dir, exist_ok=True)
                    
                    # 获取该类型的所有图片
                    images = [f for f in os.listdir(os.path.join(dataset_path, fake_type)) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if num_samples:
                        images = images[:num_samples]
                    
                    print(f"Processing {fake_type}...")
                    for img_name in tqdm(images):
                        img_path = os.path.join(dataset_path, fake_type, img_name)
                        save_path = os.path.join(type_dir, f"heatmap_{img_name}")
                        
                        try:
                            # 生成单层热力图
                            self.generate_heatmap(img_path, save_path, colormap=colormap, use_multi_layer=False)
                            # 生成多层热力图
                            self.generate_heatmap(img_path, save_path, colormap=colormap, use_multi_layer=True)
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
            else:
                # 单一图像列表
                images = [f for f in items if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if num_samples:
                    images = images[:num_samples]
                
                print(f"Processing {len(images)} images...")
                for img_name in tqdm(images):
                    img_path = os.path.join(dataset_path, img_name)
                    save_path = os.path.join(output_dir, f"heatmap_{img_name}")
                    
                    try:
                        # 生成单层热力图
                        self.generate_heatmap(img_path, save_path, colormap=colormap, use_multi_layer=False)
                        # 生成多层热力图
                        self.generate_heatmap(img_path, save_path, colormap=colormap, use_multi_layer=True)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
        else:
            # 单个图像文件
            img_name = os.path.basename(dataset_path)
            save_path = os.path.join(output_dir, f"heatmap_{img_name}")
            
            try:
                # 生成单层热力图
                self.generate_heatmap(dataset_path, save_path, colormap=colormap, use_multi_layer=False)
                # 生成多层热力图
                self.generate_heatmap(dataset_path, save_path, colormap=colormap, use_multi_layer=True)
            except Exception as e:
                print(f"Error processing {dataset_path}: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate heatmaps for fake images')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./heatmaps',
                      help='Directory to save heatmaps')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process per category')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--feature_type', type=str, default='fusion', choices=['npr', 'clip', 'fusion'],
                      help='Which feature to use for heatmap: npr, clip, or fusion')
    parser.add_argument('--colormap', type=str, default='turbo', 
                       choices=['jet', 'plasma', 'inferno', 'viridis', 'turbo', 'hot', 'cool', 'rainbow'],
                       help='颜色映射方案')
    parser.add_argument('--multi_layer', action='store_true',
                       help='使用多层融合热力图')
    parser.add_argument('--reference_style', action='store_true',
                       help='生成类似参考图风格的热力图')
    parser.add_argument('--heatmap_type', type=str, default='simple', 
                       choices=['simple', 'detailed', 'reference'],
                       help='热力图类型: simple=简单关注点, detailed=详细多层, reference=参考风格')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='热力图叠加强度 (0-1), 越大越明显')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='热力图显示阈值 (0-1), 仅显示高于此阈值的区域')
    return parser.parse_args()

def get_colormap(name):
    """获取指定的颜色映射方案"""
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'turbo': cv2.COLORMAP_TURBO,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
        'rainbow': cv2.COLORMAP_RAINBOW
    }
    return colormap_dict.get(name.lower(), cv2.COLORMAP_JET)

def generate_reference_style_heatmap(model_path, image_path, save_dir, device='cuda'):
    """生成类似参考图风格的高质量热力图"""
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型
    model = create_model(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 创建多层热力图生成器
    cam_generator = MultiLayerGradCAM(model)
    
    # 获取多个目标层
    target_layers = {}
    
    # 添加NPR层
    if hasattr(model, 'npr_extract'):
        target_layers['npr'] = model.npr_extract
    
    # 添加NPR嵌入层
    if hasattr(model, 'npr_embeddings') and len(model.npr_embeddings) > 0:
        target_layers['npr_embed'] = model.npr_embeddings[0]
    
    # 添加融合层
    if hasattr(model, 'fusion_blocks'):
        num_blocks = len(model.fusion_blocks)
        if num_blocks >= 1:
            # 只选取几个关键层，避免太多层稀释重要特征
            target_layers['fusion_first'] = model.fusion_blocks[0]
            if num_blocks >= 3:
                target_layers['fusion_last'] = model.fusion_blocks[-1]
    
    # 生成热力图
    print(f"生成参考图风格热力图，使用层: {list(target_layers.keys())}")
    cam = cam_generator(image_tensor, target_layers)
    
    # 调整回原始图像大小
    cam = cv2.resize(cam, original_size)
    
    # 应用CLAHE增强对比度
    cam = (cam * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # 增大对比度限制
    cam = clahe.apply(cam)
    
    # 锐化处理增强边缘
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    cam = cv2.filter2D(cam, -1, kernel)
    
    # 再次应用CLAHE
    cam = clahe.apply(cam)
    
    # 归一化
    cam = cam.astype(float) / 255.0
    
    # Gamma校正，突出高值区域
    cam = np.power(cam, 0.65)
    
    # 创建热力图
    heatmap = np.uint8(255 * cam)
    
    # 应用TURBO色彩映射方案
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    
    # 增强热力图颜色的饱和度和亮度
    colored_heatmap_hsv = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2HSV)
    # 增加饱和度
    colored_heatmap_hsv[:,:,1] = np.clip(colored_heatmap_hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
    # 增加亮度
    colored_heatmap_hsv[:,:,2] = np.clip(colored_heatmap_hsv[:,:,2] * 1.2, 0, 255).astype(np.uint8)
    colored_heatmap = cv2.cvtColor(colored_heatmap_hsv, cv2.COLOR_HSV2BGR)
    
    # 获取图像文件名
    img_name = os.path.basename(image_path)
    
    # 创建带透明通道的热力图 - 类似参考图
    heat_alpha = np.ones((heatmap.shape[0], heatmap.shape[1]), dtype=np.uint8) * 255
    heat_alpha[heatmap < 40] = 0  # 低响应区域设为透明
    
    # 合并RGB和Alpha通道
    colored_alpha = cv2.merge([colored_heatmap[:,:,0], colored_heatmap[:,:,1], 
                             colored_heatmap[:,:,2], heat_alpha])
    
    # 保存带透明通道的PNG (这个是最接近参考图的效果)
    ref_style_path = os.path.join(save_dir, f"reference_style_{img_name}")
    ref_style_path = ref_style_path.replace('.jpg', '.png').replace('.jpeg', '.png')
    cv2.imwrite(ref_style_path, colored_alpha)
    
    # 将原始图像转换为numpy数组
    original_image = np.array(image)
    
    # 创建热力图叠加版本
    heat_mask = np.zeros_like(original_image, dtype=np.float32)
    heat_mask[..., :] = colored_heatmap / 255.0
    
    # 基于热力图强度计算alpha
    alpha_mask = np.clip(cam * 2.5, 0.25, 0.9)
    alpha_mask = np.expand_dims(alpha_mask, axis=2)
    alpha_mask = np.repeat(alpha_mask, 3, axis=2)
    
    # 混合热力图和原始图像
    overlay = (1.0 - alpha_mask) * (original_image / 255.0) + alpha_mask * heat_mask
    overlay = (overlay * 255).astype(np.uint8)
    
    # 保存叠加结果
    overlay_path = os.path.join(save_dir, f"overlay_{img_name}")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 创建白底版本（更清晰地显示热力图）
    white_bg = np.ones_like(original_image) * 255
    white_overlay = (1.0 - alpha_mask) * (white_bg / 255.0) + alpha_mask * heat_mask
    white_overlay = (white_overlay * 255).astype(np.uint8)
    
    # 保存白底热力图
    white_path = os.path.join(save_dir, f"white_bg_{img_name}")
    cv2.imwrite(white_path, cv2.cvtColor(white_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"参考风格热力图已保存至 {save_dir}")
    return ref_style_path

def generate_simple_heatmap(model_path, image_path, save_dir, device='cuda', alpha=0.7, threshold=0.3):
    """生成简洁的热力图，只突出显示模型关注的关键点"""
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型
    model = create_model(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 创建多层热力图生成器
    cam_generator = MultiLayerGradCAM(model)
    
    # 获取多个目标层
    target_layers = {}
    
    # 添加NPR层和融合层
    if hasattr(model, 'npr_extract'):
        target_layers['npr'] = model.npr_extract
    
    if hasattr(model, 'fusion_blocks'):
        num_blocks = len(model.fusion_blocks)
        if num_blocks >= 1:
            middle_idx = num_blocks // 2
            target_layers['fusion'] = model.fusion_blocks[middle_idx]
    
    # 生成热力图
    cam = cam_generator(image_tensor, target_layers)
    
    # 调整回原始图像大小
    cam = cv2.resize(cam, original_size)
    
    # 增强对比度
    cam = (cam * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cam = clahe.apply(cam)
    
    # 平滑处理
    cam = cv2.GaussianBlur(cam, (5, 5), 1.0)
    
    # 归一化
    cam = cam.astype(float) / 255.0
    
    # 只保留高响应区域，其他区域降低显示强度
    cam = np.power(cam, 0.7)  # gamma校正
    
    # 创建热力图
    heatmap = np.uint8(255 * cam)
    
    # 获取图像文件名
    img_name = os.path.basename(image_path)
    
    # 将原始图像转换为numpy数组
    original_image = np.array(image)
    
    # 应用热力图颜色映射 - 使用JET映射，类似参考图
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 增强颜色的饱和度
    colored_heatmap_hsv = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2HSV)
    colored_heatmap_hsv[:,:,1] = np.clip(colored_heatmap_hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
    colored_heatmap = cv2.cvtColor(colored_heatmap_hsv, cv2.COLOR_HSV2BGR)
    
    # 类似用户提供的图片的热力图效果，使原图更清晰
    # 创建热力图遮罩
    heat_mask = np.zeros_like(original_image, dtype=np.float32)
    heat_mask[..., :] = colored_heatmap / 255.0
    
    # 只在高响应区域显示热力图
    high_response_mask = cam > threshold
    
    # 创建渐变过渡的透明度掩码，而不是硬边界
    # 对cam进行高斯模糊，创造平滑渐变效果
    smooth_cam = cv2.GaussianBlur(cam, (15, 15), 0)
    
    # 使用平滑的cam值作为透明度，实现渐变效果
    alpha_map = np.clip((smooth_cam - threshold) / (1.0 - threshold) * alpha, 0.0, alpha)
    
    # 扩展到3通道
    alpha_mask = np.expand_dims(alpha_map, axis=2)
    alpha_mask = np.repeat(alpha_mask, 3, axis=2)
    
    # 混合热力图和原始图像 - 使用渐变透明度
    attention_overlay = original_image.copy() / 255.0
    
    # 对原图进行轻微提亮处理，使原图细节更清晰
    # 使用自适应直方图均衡化处理原图，使细节更清晰
    original_image_enhanced = original_image.copy()
    original_image_gray = cv2.cvtColor(original_image_enhanced, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    original_image_gray = clahe.apply(original_image_gray)
    
    # 只提升亮度，不改变颜色
    original_hsv = cv2.cvtColor(original_image_enhanced, cv2.COLOR_RGB2HSV)
    original_hsv[:,:,2] = np.clip(original_hsv[:,:,2] * 1.1, 0, 255).astype(np.uint8)
    original_image_enhanced = cv2.cvtColor(original_hsv, cv2.COLOR_HSV2RGB)
    
    # 混合热力图和增强后的原图
    attention_overlay = original_image_enhanced / 255.0
    attention_overlay = (1.0 - alpha_mask) * attention_overlay + alpha_mask * heat_mask
    attention_overlay = (attention_overlay * 255).astype(np.uint8)
    
    # 保存关注点热力图
    attention_path = os.path.join(save_dir, f"attention_{img_name}")
    cv2.imwrite(attention_path, cv2.cvtColor(attention_overlay, cv2.COLOR_RGB2BGR))
    
    # 创建直接叠加版本，但使用较低的alpha以保持原图清晰
    overlay = cv2.addWeighted(original_image_enhanced, 0.7, colored_heatmap, 0.3, 0)
    
    # 保存简单叠加版本
    simple_path = os.path.join(save_dir, f"simple_{img_name}")
    cv2.imwrite(simple_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 创建更接近参考图的版本，但保持原图更清晰
    # 应用高斯模糊到高响应掩码，创建平滑边缘
    mask_blur = cv2.GaussianBlur(high_response_mask.astype(np.float32), (9, 9), 0)
    mask_blur = np.expand_dims(mask_blur, axis=2)
    mask_blur = np.repeat(mask_blur, 3, axis=2)
    
    # 使用平滑掩码创建更柔和的混合效果
    clean_overlay = original_image_enhanced.copy() / 255.0
    clean_overlay = (1.0 - mask_blur * alpha) * clean_overlay + (mask_blur * alpha) * heat_mask
    clean_overlay = (clean_overlay * 255).astype(np.uint8)
    
    # 保存清晰版热力图
    clean_path = os.path.join(save_dir, f"clean_{img_name}")
    cv2.imwrite(clean_path, cv2.cvtColor(clean_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"简洁热力图已保存至 {save_dir}")
    return attention_path

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 获取指定的颜色映射方案
    colormap = get_colormap(args.colormap)
    
    # 根据热力图类型选择处理方式
    if args.heatmap_type == 'simple':
        # 检查输入是文件还是目录
        if os.path.isfile(args.dataset_path):
            # 处理单个图像
            generate_simple_heatmap(
                args.model_path, 
                args.dataset_path, 
                args.output_dir, 
                args.device,
                alpha=args.alpha,
                threshold=args.threshold
            )
        else:
            # 处理目录结构，保持输出与输入一致
            if os.path.isdir(args.dataset_path):
                # 检查是否有子目录
                has_subdirs = any(os.path.isdir(os.path.join(args.dataset_path, item)) 
                               for item in os.listdir(args.dataset_path))
                
                if has_subdirs:
                    # 有子目录，处理每个子目录下的图像
                    for subdir in os.listdir(args.dataset_path):
                        subdir_path = os.path.join(args.dataset_path, subdir)
                        if os.path.isdir(subdir_path):
                            # 为每个子目录创建对应的输出目录
                            output_subdir = os.path.join(args.output_dir, subdir)
                            os.makedirs(output_subdir, exist_ok=True)
                            
                            # 获取子目录下的所有图像
                            image_files = [f for f in os.listdir(subdir_path)
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            
                            # 限制处理的图像数量
                            if args.num_samples and len(image_files) > args.num_samples:
                                image_files = image_files[:args.num_samples]
                            
                            # 处理图像
                            for img_name in tqdm(image_files, desc=f"处理 {subdir} 中的图像"):
                                img_path = os.path.join(subdir_path, img_name)
                                try:
                                    generate_simple_heatmap(
                                        args.model_path,
                                        img_path,
                                        output_subdir,
                                        args.device,
                                        alpha=args.alpha,
                                        threshold=args.threshold
                                    )
                                except Exception as e:
                                    print(f"处理 {img_path} 时出错: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                else:
                    # 没有子目录，直接处理图像
                    image_files = [f for f in os.listdir(args.dataset_path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    # 限制处理的图像数量
                    if args.num_samples and len(image_files) > args.num_samples:
                        image_files = image_files[:args.num_samples]
                    
                    # 处理图像
                    for img_name in tqdm(image_files, desc="处理图像"):
                        img_path = os.path.join(args.dataset_path, img_name)
                        try:
                            generate_simple_heatmap(
                                args.model_path,
                                img_path,
                                args.output_dir,
                                args.device,
                                alpha=args.alpha,
                                threshold=args.threshold
                            )
                        except Exception as e:
                            print(f"处理 {img_path} 时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
        return
    
    # 生成参考风格热力图
    elif args.heatmap_type == 'reference' or args.reference_style:
        # 检查输入是文件还是目录
        if os.path.isfile(args.dataset_path):
            # 处理单个图像
            generate_reference_style_heatmap(
                args.model_path, 
                args.dataset_path, 
                args.output_dir, 
                args.device
            )
        else:
            # 处理目录结构，保持输出与输入一致
            if os.path.isdir(args.dataset_path):
                # 检查是否有子目录
                has_subdirs = any(os.path.isdir(os.path.join(args.dataset_path, item)) 
                               for item in os.listdir(args.dataset_path))
                
                if has_subdirs:
                    # 有子目录，处理每个子目录下的图像
                    for subdir in os.listdir(args.dataset_path):
                        subdir_path = os.path.join(args.dataset_path, subdir)
                        if os.path.isdir(subdir_path):
                            # 为每个子目录创建对应的输出目录
                            output_subdir = os.path.join(args.output_dir, subdir)
                            os.makedirs(output_subdir, exist_ok=True)
                            
                            # 获取子目录下的所有图像
                            image_files = [f for f in os.listdir(subdir_path)
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            
                            # 限制处理的图像数量
                            if args.num_samples and len(image_files) > args.num_samples:
                                image_files = image_files[:args.num_samples]
                            
                            # 处理图像
                            for img_name in tqdm(image_files, desc=f"处理 {subdir} 中的图像"):
                                img_path = os.path.join(subdir_path, img_name)
                                try:
                                    generate_reference_style_heatmap(
                                        args.model_path,
                                        img_path,
                                        output_subdir,
                                        args.device
                                    )
                                except Exception as e:
                                    print(f"处理 {img_path} 时出错: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                else:
                    # 没有子目录，直接处理图像
                    image_files = [f for f in os.listdir(args.dataset_path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    # 限制处理的图像数量
                    if args.num_samples and len(image_files) > args.num_samples:
                        image_files = image_files[:args.num_samples]
                    
                    # 处理图像
                    for img_name in tqdm(image_files, desc="处理图像"):
                        img_path = os.path.join(args.dataset_path, img_name)
                        try:
                            generate_reference_style_heatmap(
                                args.model_path,
                                img_path,
                                args.output_dir,
                                args.device
                            )
                        except Exception as e:
                            print(f"处理 {img_path} 时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
        return
    
    # 生成详细的多层热力图 - HeatmapGenerator 已经支持目录结构
    elif args.heatmap_type == 'detailed' or args.multi_layer:
        # 使用标准热力图生成器（多层模式）
        generator = HeatmapGenerator(args.model_path, device=args.device, feature_type=args.feature_type)
        generator.process_dataset(args.dataset_path, args.output_dir, args.num_samples, colormap=colormap)
        return
    
    # 默认使用HeatmapGenerator，它已经支持保持目录结构
    else:
        print("使用标准热力图生成器")
        generator = HeatmapGenerator(args.model_path, device=args.device, feature_type=args.feature_type)
        generator.process_dataset(args.dataset_path, args.output_dir, args.num_samples, colormap=colormap)

if __name__ == "__main__":
    main() 