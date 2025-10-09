import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from networks.vit_npr import create_model
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

class FeatureVisualizer:
    def __init__(self, model_path, device='cuda'):
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = create_model(num_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型已从 {model_path} 加载")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 特征存储
        self.hooks = []
        self.features = {}
    
    def _register_hooks(self):
        """注册钩子函数以捕获各种特征"""
        # 清除之前的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
        
        # NPR特征提取器
        def npr_hook(module, input, output):
            self.features['npr'] = output.detach()
        
        hook = self.model.npr_extract.register_forward_hook(npr_hook)
        self.hooks.append(hook)
        
        # CLIP特征钩子 - 捕获CLIP视觉特征
        def clip_conv_hook(module, input, output):
            # 获取CLIP的卷积特征
            self.features['clip_conv'] = output.detach()
        
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'conv1'):
            hook = self.model.visual.conv1.register_forward_hook(clip_conv_hook)
            self.hooks.append(hook)
        
        # 捕获CLIP的Transformer特征
        def clip_transformer_hook(module, input, output):
            if 'clip_transformer' not in self.features:
                self.features['clip_transformer'] = []
            self.features['clip_transformer'].append(output.detach())
        
        # 尝试注册CLIP Transformer钩子
        try:
            for i, block in enumerate(self.model.visual.transformer.resblocks[-4:]):
                hook = block.register_forward_hook(clip_transformer_hook)
                self.hooks.append(hook)
        except (AttributeError, IndexError):
            print("无法访问CLIP Transformer块，跳过")
            
        # NPR嵌入特征钩子
        for idx, embed in enumerate(self.model.npr_embeddings):
            def embed_hook(idx):
                def hook(module, input, output):
                    self.features[f'npr_embed_{idx}'] = output.detach()
                return hook
            
            h = embed.register_forward_hook(embed_hook(idx))
            self.hooks.append(h)
        
        # 融合模块钩子
        for idx, block in enumerate(self.model.fusion_blocks):
            def fusion_block_hook(idx):
                def hook(module, input, output):
                    self.features[f'fusion_block_{idx}'] = output.detach()
                return hook
            
            h = block.register_forward_hook(fusion_block_hook(idx))
            self.hooks.append(h)
        
        # 最终融合层钩子
        def final_fusion_hook(module, input, output):
            self.features['final_fusion'] = output.detach()
        
        hook = self.model.norm.register_forward_hook(final_fusion_hook)
        self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _normalize_map(self, feature_map):
        """归一化特征图到0-255范围"""
        f_min = feature_map.min()
        f_max = feature_map.max()
        if f_max > f_min:
            return ((feature_map - f_min) / (f_max - f_min) * 255).astype(np.uint8)
        else:
            return np.zeros_like(feature_map, dtype=np.uint8)
    
    def _get_token_features(self, features):
        """处理token类特征，支持非完美平方数的token数量
        
        Args:
            features: 形状为 [N, C] 的特征矩阵
            
        Returns:
            token_map: 处理后的特征图
            success: 是否成功处理
        """
        # 确保有足够的token
        if features.shape[0] <= 1:
            print(f"token特征形状不足: {features.shape}")
            return None, False
            
        # 移除CLS token并获取patch tokens
        patch_features = features[1:, :]  # 移除CLS token
        
        # 计算token特征的平均值
        token_mean = np.mean(patch_features, axis=1)
        
        # 处理方式1：尝试完美平方形重塑
        total_tokens = token_mean.shape[0]
        size = int(np.sqrt(total_tokens))
        
        if size * size == total_tokens:
            # 是完美平方数
            token_map = token_mean.reshape(size, size)
            token_norm = self._normalize_map(token_map)
            return cv2.resize(token_norm, (256, 256), interpolation=cv2.INTER_NEAREST), True
        
        # 处理方式2：寻找最优矩形布局
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
            token_norm = self._normalize_map(token_map)
            return cv2.resize(token_norm, (256, 256), interpolation=cv2.INTER_NEAREST), True
        
        # 处理方式3：填充到最近的平方数
        next_square = int(np.ceil(np.sqrt(total_tokens))) ** 2
        padding = np.zeros(next_square - total_tokens)
        padded_tokens = np.concatenate([token_mean, padding])
        
        size = int(np.sqrt(next_square))
        token_map = padded_tokens.reshape(size, size)
        token_norm = self._normalize_map(token_map)
        
        print(f"非完美平方数的token: {total_tokens}，填充到 {size}x{size}")
        return cv2.resize(token_norm, (256, 256), interpolation=cv2.INTER_NEAREST), True
    
    def _process_channel_features(self, features, name, output_dir, image_name):
        """处理通道类特征 [C,H,W]"""
        try:
            # 将特征转换为可视化格式 [C,H,W] -> [H,W,C]
            feature_map = features.permute(1, 2, 0).numpy()
            # 计算通道平均值获得灰度图
            feature_gray = np.mean(feature_map, axis=2)
            
            # 边缘检测增强
            sobel_x = cv2.Sobel(feature_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(feature_gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 归一化
            edge_norm = self._normalize_map(edge_map)
            gray_norm = self._normalize_map(feature_gray)
            
            # 确保尺寸为256x256
            if edge_norm.shape[0] != 256 or edge_norm.shape[1] != 256:
                edge_norm = cv2.resize(edge_norm, (256, 256))
            if gray_norm.shape[0] != 256 or gray_norm.shape[1] != 256:
                gray_norm = cv2.resize(gray_norm, (256, 256))
            
            # 保存边缘图和灰度图
            cv2.imwrite(os.path.join(output_dir, f"{name}_edge_{image_name}"), edge_norm)
            cv2.imwrite(os.path.join(output_dir, f"{name}_gray_{image_name}"), gray_norm)
            
            return edge_norm, gray_norm
        except Exception as e:
            print(f"处理{name}特征时出错: {str(e)}")
            # 返回空白图像，避免后续处理出错
            blank = np.zeros((256, 256), dtype=np.uint8)
            return blank, blank
    
    def _process_token_features(self, features, name, output_dir, image_name):
        """处理token类特征 [N,C]"""
        try:
            # 使用改进的token特征处理函数
            token_map, success = self._get_token_features(features)
            
            if not success or token_map is None:
                print(f"处理{name}的token特征失败")
                blank = np.zeros((256, 256), dtype=np.uint8)
                return blank, blank
            
            # 保存token图像
            cv2.imwrite(os.path.join(output_dir, f"{name}_token_{image_name}"), token_map)
            
            # 为保持一致性，也保存一个gray版本
            cv2.imwrite(os.path.join(output_dir, f"{name}_gray_{image_name}"), token_map)
            
            # 为token特征也添加边缘增强处理，保持与通道特征的一致性
            # 应用Sobel边缘检测
            sobel_x = cv2.Sobel(token_map, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(token_map, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 归一化
            edge_norm = self._normalize_map(edge_map)
            
            # 保存边缘特征图
            cv2.imwrite(os.path.join(output_dir, f"{name}_edge_{image_name}"), edge_norm)
            
            return token_map, edge_norm
                
        except Exception as e:
            print(f"处理token特征时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空白图像，避免后续处理出错
            blank = np.zeros((256, 256), dtype=np.uint8)
            return blank, blank
    
    def visualize_image(self, image_path, output_dir, visualize_type='overlay'):
        """可视化单个图像的特征"""
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        
        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return None
        
        # 注册钩子并前向传播
        self._register_hooks()
        with torch.no_grad():
            try:
                self.model(input_tensor)
            except Exception as e:
                print(f"模型前向传播时出错: {str(e)}")
                self._remove_hooks()
                return None
        
        # 保存原始图像
        orig_img = np.array(image.resize((256, 256)))
        cv2.imwrite(os.path.join(output_dir, f"original_{image_name}"), 
                  cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        
        # 创建BGR格式的原图，用于叠加
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        # 处理NPR特征
        npr_edge = None
        npr_gray = None
        npr_heatmap = None
        
        if 'npr' in self.features:
            try:
                npr_feature = self.features['npr'].squeeze().cpu()
                npr_edge, npr_gray = self._process_channel_features(
                    npr_feature, 'npr', output_dir, image_name)
                
                # 创建NPR热力图
                npr_heatmap = cv2.applyColorMap(npr_edge, cv2.COLORMAP_JET)
                
                # 确保尺寸匹配
                if npr_heatmap.shape[:2] != orig_img_bgr.shape[:2]:
                    npr_heatmap = cv2.resize(npr_heatmap, (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
                
                if visualize_type == 'overlay':
                    # 叠加热力图到原图
                    overlay_npr = cv2.addWeighted(orig_img_bgr, 0.6, npr_heatmap, 0.4, 0)
                    cv2.imwrite(os.path.join(output_dir, f"overlay_npr_{image_name}"), overlay_npr)
                else:
                    # 只保存热力图
                    cv2.imwrite(os.path.join(output_dir, f"heatmap_npr_{image_name}"), npr_heatmap)
            except Exception as e:
                print(f"处理NPR特征时出错: {str(e)}")
        
        # 处理CLIP特征
        clip_edge = None
        clip_gray = None
        clip_heatmap = None
        
        if 'clip_conv' in self.features:
            try:
                clip_feature = self.features['clip_conv'].squeeze().cpu()
                clip_edge, clip_gray = self._process_channel_features(
                    clip_feature, 'clip', output_dir, image_name)
                
                # 创建CLIP热力图
                clip_heatmap = cv2.applyColorMap(clip_edge, cv2.COLORMAP_PLASMA)
                
                # 确保尺寸匹配
                if clip_heatmap.shape[:2] != orig_img_bgr.shape[:2]:
                    clip_heatmap = cv2.resize(clip_heatmap, (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
                
                if visualize_type == 'overlay':
                    # 叠加热力图到原图
                    overlay_clip = cv2.addWeighted(orig_img_bgr, 0.6, clip_heatmap, 0.4, 0)
                    cv2.imwrite(os.path.join(output_dir, f"overlay_clip_{image_name}"), overlay_clip)
                else:
                    cv2.imwrite(os.path.join(output_dir, f"heatmap_clip_{image_name}"), clip_heatmap)
            except Exception as e:
                print(f"处理CLIP特征时出错: {str(e)}")
        
        # 处理最终融合特征
        fusion_map = None
        fusion_edge = None
        fusion_heatmap = None
        
        if 'final_fusion' in self.features:
            try:
                fusion_feature = self.features['final_fusion'].squeeze().cpu().numpy()
                
                # 调试信息
                print(f"融合特征形状: {fusion_feature.shape}")
                
                # 使用改进的token特征处理函数
                fusion_map, fusion_edge = self._process_token_features(
                    fusion_feature, 'fusion', output_dir, image_name)
                
                if fusion_map is not None:
                    # 增强对比度和清晰度
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    fusion_enhanced = clahe.apply(fusion_map)
                    
                    # 锐化处理
                    fusion_sharpened = cv2.addWeighted(fusion_enhanced, 1.5, 
                                                   cv2.GaussianBlur(fusion_enhanced, (5, 5), 0), -0.5, 0)
                    
                    # 确保尺寸匹配
                    if fusion_sharpened.shape[:2] != orig_img_bgr.shape[:2]:
                        fusion_sharpened = cv2.resize(fusion_sharpened, 
                                                  (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
                    
                    # 使用边缘增强图创建融合特征热力图 - 使用TURBO色彩映射
                    if fusion_edge is not None:
                        # 确保尺寸匹配
                        if fusion_edge.shape[:2] != orig_img_bgr.shape[:2]:
                            fusion_edge = cv2.resize(fusion_edge, 
                                                 (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
                        
                        # 应用CLAHE和锐化处理
                        fusion_edge_enhanced = clahe.apply(fusion_edge)
                        fusion_edge_sharpened = cv2.addWeighted(fusion_edge_enhanced, 1.5, 
                                                           cv2.GaussianBlur(fusion_edge_enhanced, (5, 5), 0), -0.5, 0)
                        
                        # 使用更好的色彩映射方案
                        fusion_heatmap = cv2.applyColorMap(fusion_edge_sharpened, cv2.COLORMAP_TURBO)
                    else:
                        # 如果边缘图不可用，使用增强的token map
                        fusion_heatmap = cv2.applyColorMap(fusion_sharpened, cv2.COLORMAP_TURBO)
                    
                    # 增强热力图的饱和度
                    fusion_heatmap_hsv = cv2.cvtColor(fusion_heatmap, cv2.COLOR_BGR2HSV)
                    fusion_heatmap_hsv[:,:,1] = np.clip(fusion_heatmap_hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
                    fusion_heatmap = cv2.cvtColor(fusion_heatmap_hsv, cv2.COLOR_HSV2BGR)
                    
                    # 保存标准热力图
                    cv2.imwrite(os.path.join(output_dir, f"heatmap_fusion_{image_name}"), fusion_heatmap)
                    
                    if visualize_type == 'overlay':
                        # 叠加热力图到原图 - 更强调热力图的效果
                        overlay_fusion = cv2.addWeighted(orig_img_bgr, 0.3, fusion_heatmap, 0.7, 0)
                        cv2.imwrite(os.path.join(output_dir, f"overlay_fusion_{image_name}"), overlay_fusion)
                else:
                    print("融合特征处理返回为None，检查token特征处理逻辑")
            except Exception as e:
                print(f"处理融合特征时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("无法找到'final_fusion'特征，请检查钩子注册")
            # 尝试找到类似的融合特征
            fusion_keys = [k for k in self.features.keys() if 'fusion' in k]
            if fusion_keys:
                print(f"找到的融合相关特征: {fusion_keys}")
        
        # 创建混合特征可视化 - 组合NPR和融合特征
        hybrid_heatmap = None
        
        if npr_edge is not None and fusion_edge is not None:
            try:
                # 确保尺寸匹配
                if npr_edge.shape != fusion_edge.shape:
                    print(f"调整尺寸匹配: npr_edge={npr_edge.shape}, fusion_edge={fusion_edge.shape}")
                    npr_edge_resized = cv2.resize(npr_edge, (fusion_edge.shape[1], fusion_edge.shape[0]))
                else:
                    npr_edge_resized = npr_edge
                
                # 改进混合特征处理方式 - 使用自适应权重而非固定权重
                # 计算两个特征图的清晰度/对比度
                npr_contrast = np.std(npr_edge_resized)
                fusion_contrast = np.std(fusion_edge)
                
                # 根据对比度动态调整权重
                weight_npr = npr_contrast / (npr_contrast + fusion_contrast + 1e-6)
                weight_fusion = 1 - weight_npr
                
                # 增强对比度和锐度
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                npr_enhanced = clahe.apply(npr_edge_resized)
                fusion_enhanced = clahe.apply(fusion_edge)
                
                # 分别应用边缘增强
                npr_sharpened = cv2.addWeighted(npr_enhanced, 1.5, cv2.GaussianBlur(npr_enhanced, (5, 5), 0), -0.5, 0)
                fusion_sharpened = cv2.addWeighted(fusion_enhanced, 1.5, cv2.GaussianBlur(fusion_enhanced, (5, 5), 0), -0.5, 0)
                
                # 创建混合特征图 - 使用增强后的特征和自适应权重
                hybrid_map = cv2.addWeighted(npr_sharpened, 0.6, fusion_sharpened, 0.4, 0)
                
                # 再次提高对比度
                hybrid_map = clahe.apply(hybrid_map)
                
                # 应用更鲜明的色彩映射
                hybrid_heatmap = cv2.applyColorMap(hybrid_map, cv2.COLORMAP_JET)
                
                # 确保与原图尺寸匹配
                if hybrid_heatmap.shape[:2] != orig_img_bgr.shape[:2]:
                    hybrid_heatmap = cv2.resize(hybrid_heatmap, (orig_img_bgr.shape[1], orig_img_bgr.shape[0]))
                
                # 保存标准热力图
                cv2.imwrite(os.path.join(output_dir, f"heatmap_hybrid_{image_name}"), hybrid_heatmap)
                
                # 为了更清晰的展示，降低原图权重，增加热力图权重
                if visualize_type == 'overlay':
                    # 叠加混合热力图到原图 - 调整透明度
                    overlay_hybrid = cv2.addWeighted(orig_img_bgr, 0.4, hybrid_heatmap, 0.6, 0)
                    # 额外保存一个增强版叠加图
                    cv2.imwrite(os.path.join(output_dir, f"overlay_hybrid_{image_name}"), overlay_hybrid)
                    
                    # 保存一个更强调热力图的版本
                    overlay_hybrid_enhanced = cv2.addWeighted(orig_img_bgr, 0.3, hybrid_heatmap, 0.7, 0)
                    cv2.imwrite(os.path.join(output_dir, f"overlay_hybrid_enhanced_{image_name}"), overlay_hybrid_enhanced)
            except Exception as e:
                print(f"创建混合特征图时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 创建特征对比图 - 将多个特征图组合在一起
        try:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 显示原图
            axes[0].imshow(image.resize((256, 256)))
            axes[0].set_title("原始图像", fontsize=14)
            axes[0].axis('off')
            
            # 显示NPR特征
            if npr_heatmap is not None:
                axes[1].imshow(cv2.cvtColor(npr_heatmap, cv2.COLOR_BGR2RGB))
                axes[1].set_title("NPR特征", fontsize=14)
            else:
                axes[1].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                axes[1].set_title("NPR特征 (不可用)", fontsize=14)
            axes[1].axis('off')
            
            # 显示融合特征
            if fusion_heatmap is not None:
                axes[2].imshow(cv2.cvtColor(fusion_heatmap, cv2.COLOR_BGR2RGB))
                axes[2].set_title("融合特征", fontsize=14)
            else:
                axes[2].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                axes[2].set_title("融合特征 (不可用)", fontsize=14)
            axes[2].axis('off')
            
            # 显示混合特征
            if hybrid_heatmap is not None:
                axes[3].imshow(cv2.cvtColor(hybrid_heatmap, cv2.COLOR_BGR2RGB))
                axes[3].set_title("混合特征", fontsize=14)
            else:
                axes[3].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                axes[3].set_title("混合特征 (不可用)", fontsize=14)
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"features_comparison_{image_name}"), 
                       dpi=200, bbox_inches='tight')  # 提高DPI以获得更清晰的图像
            plt.close()
            
            # 创建第二个对比图 - 显示不同特征的叠加效果
            if visualize_type == 'overlay':
                try:
                    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # 显示原图
                    axes2[0, 0].imshow(image.resize((256, 256)))
                    axes2[0, 0].set_title("原始图像", fontsize=14)
                    axes2[0, 0].axis('off')
                    
                    # 显示NPR叠加
                    if 'overlay_npr_' + image_name in os.listdir(output_dir):
                        npr_overlay = cv2.imread(os.path.join(output_dir, f"overlay_npr_{image_name}"))
                        axes2[0, 1].imshow(cv2.cvtColor(npr_overlay, cv2.COLOR_BGR2RGB))
                        axes2[0, 1].set_title("NPR特征叠加", fontsize=14)
                    else:
                        axes2[0, 1].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                        axes2[0, 1].set_title("NPR特征叠加 (不可用)", fontsize=14)
                    axes2[0, 1].axis('off')
                    
                    # 显示融合特征叠加
                    if 'overlay_fusion_' + image_name in os.listdir(output_dir):
                        fusion_overlay = cv2.imread(os.path.join(output_dir, f"overlay_fusion_{image_name}"))
                        axes2[1, 0].imshow(cv2.cvtColor(fusion_overlay, cv2.COLOR_BGR2RGB))
                        axes2[1, 0].set_title("融合特征叠加", fontsize=14)
                    else:
                        axes2[1, 0].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                        axes2[1, 0].set_title("融合特征叠加 (不可用)", fontsize=14)
                    axes2[1, 0].axis('off')
                    
                    # 显示混合特征叠加
                    if 'overlay_hybrid_enhanced_' + image_name in os.listdir(output_dir):
                        hybrid_overlay = cv2.imread(os.path.join(output_dir, f"overlay_hybrid_enhanced_{image_name}"))
                        axes2[1, 1].imshow(cv2.cvtColor(hybrid_overlay, cv2.COLOR_BGR2RGB))
                        axes2[1, 1].set_title("增强混合特征叠加", fontsize=14)
                    else:
                        axes2[1, 1].imshow(np.zeros((256, 256, 3), dtype=np.uint8))
                        axes2[1, 1].set_title("增强混合特征叠加 (不可用)", fontsize=14)
                    axes2[1, 1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"overlays_comparison_{image_name}"), 
                               dpi=200, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"创建叠加对比图时出错: {str(e)}")
        except Exception as e:
            print(f"创建特征对比图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 移除钩子
        self._remove_hooks()
        
        generated_files = []
        for filename in os.listdir(output_dir):
            if image_name in filename:
                generated_files.append(filename)
        
        print(f"为图像 {image_name} 生成了以下文件: {', '.join(generated_files)}")
        
        return os.path.join(output_dir, f"features_comparison_{image_name}")
    
    def visualize_directory(self, input_dir, output_dir, max_images=100, visualize_type='overlay'):
        """处理整个目录的图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, file))
        
        # 限制图像数量
        if max_images > 0 and len(image_files) > max_images:
            print(f"处理前 {max_images} 张图像(共 {len(image_files)} 张)")
            image_files = image_files[:max_images]
        
        # 处理每张图像
        results = []
        for img_path in tqdm(image_files, desc="生成特征图"):
            output_path = self.visualize_image(img_path, output_dir, visualize_type)
            if output_path:
                results.append(output_path)
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description='生成深度伪造检测模型的混合特征可视化')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='模型路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图像或目录路径')
    parser.add_argument('--output_dir', type=str, default='./feature_vis',
                      help='输出目录')
    parser.add_argument('--max_images', type=int, default=100,
                      help='处理的最大图像数量(用于目录)')
    parser.add_argument('--type', type=str, default='overlay', choices=['overlay', 'heatmap'],
                      help='可视化类型: overlay=叠加到原图, heatmap=只显示热力图')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建可视化器
    visualizer = FeatureVisualizer(args.model_path)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 处理单个文件
        print(f"处理图像: {args.input}")
        output_path = visualizer.visualize_image(args.input, args.output_dir, args.type)
        if output_path:
            print(f"特征图已保存至: {args.output_dir}")
    else:
        # 处理目录
        print(f"处理目录: {args.input}")
        output_paths = visualizer.visualize_directory(args.input, args.output_dir, 
                                                  args.max_images, args.type)
        print(f"已处理 {len(output_paths)} 张图像，特征图保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 