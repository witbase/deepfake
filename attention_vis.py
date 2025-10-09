import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from networks.vit_npr import create_model
import argparse
from tqdm import tqdm

class AttentionVisualizer:
    def __init__(self, model_path, device='cuda'):
        """初始化注意力可视化器"""
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
        
        # 保存特征和注意力图
        self.hooks = []
        self.attention_maps = {}
        
    def _register_hooks(self):
        """为模型各层注册钩子以捕获注意力图"""
        # 清除之前的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}
        
        # 为Fusion块注册钩子
        for idx, block in enumerate(self.model.fusion_blocks):
            # 获取Attention层 - SwinTransformerBlock中的attn
            attn_module = block.attn
            
            def attention_hook(idx):
                def hook(module, input, output):
                    # 注册前向钩子来捕获注意力得分
                    # 对于SwinTransformerBlock，我们需要捕获注意力权重
                    if hasattr(module, 'attn'):
                        self.attention_maps[f'block_{idx}'] = output
                return hook
            
            handle = block.register_forward_hook(attention_hook(idx))
            self.hooks.append(handle)
            
        # 为多头注意力模块注册钩子以捕获注意力权重
        def get_submodules(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, torch.nn.Module):
                    if isinstance(child, type(self.model.fusion_blocks[0].attn)):
                        # 这是我们感兴趣的MultiHeadAttention模块
                        def attn_weight_hook(name):
                            def hook(module, input, output):
                                # 存储注意力权重
                                self.attention_maps[name] = output
                            return hook
                        
                        handle = child.register_forward_hook(attn_weight_hook(full_name))
                        self.hooks.append(handle)
                    else:
                        # 递归检查子模块
                        get_submodules(child, full_name)
        
        # 开始递归注册注意力钩子
        get_submodules(self.model)
        
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def visualize_attention(self, image_path, output_dir):
        """可视化图像的注意力图"""
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
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
            output = self.model(input_tensor)
        
        # 保存原始图像作为参考
        resized_image = image.resize((256, 256))
        resized_image.save(os.path.join(output_dir, f"original_{image_name}"))
        
        # 处理每个捕获的注意力图
        attention_outputs = []
        for name, attention in self.attention_maps.items():
            try:
                # 处理注意力输出 - 这将根据模型架构略有不同
                if isinstance(attention, tuple):
                    # 有些模块可能返回元组，我们需要选择注意力权重
                    attention_weights = attention[0]  # 可能需要调整索引
                elif hasattr(attention, 'shape') and len(attention.shape) == 4:
                    # B, H, N, N 格式的注意力权重
                    attention_weights = attention
                else:
                    # 其他格式，可能需要根据实际情况调整
                    continue
                
                # 移到CPU并转换为numpy
                attn = attention_weights.squeeze().cpu().numpy()
                
                # 如果是多头注意力，取平均值
                if len(attn.shape) == 3:  # [heads, tokens, tokens]
                    attn = attn.mean(axis=0)  # 平均所有头
                
                # 只保留与CLS token的注意力 (如果有CLS token)
                if attn.shape[0] > 1:
                    cls_attention = attn[0, 1:]  # CLS对其他token的注意力
                else:
                    cls_attention = attn.flatten()
                
                # 重塑为图像格式 (假设是方形的)
                size = int(np.sqrt(cls_attention.shape[0]))
                if size * size != cls_attention.shape[0]:
                    print(f"无法将注意力权重重塑为正方形图像: {cls_attention.shape}")
                    continue
                    
                attention_map = cls_attention.reshape(size, size)
                
                # 放大注意力图到256x256
                attention_resized = cv2.resize(attention_map, (256, 256), 
                                            interpolation=cv2.INTER_LINEAR)
                
                # 归一化到0-1
                attention_norm = (attention_resized - attention_resized.min()) / \
                               (attention_resized.max() - attention_resized.min() + 1e-8)
                
                # 转换为0-255的热力图
                heatmap = np.uint8(attention_norm * 255)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # 保存热力图
                cv2.imwrite(os.path.join(output_dir, f"{name}_heatmap_{image_name}"), heatmap)
                
                # 创建原图与热力图的混合
                resized_np = np.array(resized_image)
                resized_np = cv2.cvtColor(resized_np, cv2.COLOR_RGB2BGR)
                
                # 叠加热力图
                overlay = cv2.addWeighted(resized_np, 0.6, heatmap, 0.4, 0)
                cv2.imwrite(os.path.join(output_dir, f"{name}_overlay_{image_name}"), overlay)
                
                attention_outputs.append(os.path.join(output_dir, f"{name}_overlay_{image_name}"))
            except Exception as e:
                print(f"处理注意力图 {name} 时出错: {str(e)}")
        
        # 移除钩子
        self._remove_hooks()
        
        # 创建合并图像 - 将所有注意力图放在一张图上
        if len(attention_outputs) > 0:
            try:
                fig, axes = plt.subplots(1, len(attention_outputs) + 1, figsize=(5*(len(attention_outputs) + 1), 5))
                
                # 显示原图
                axes[0].imshow(np.array(resized_image))
                axes[0].set_title("原始图像")
                axes[0].axis('off')
                
                # 显示每个注意力图
                for i, att_path in enumerate(attention_outputs):
                    overlay = cv2.imread(att_path)
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    axes[i+1].imshow(overlay)
                    layer_name = os.path.basename(att_path).split('_')[0]
                    axes[i+1].set_title(f"{layer_name} 注意力")
                    axes[i+1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"all_attention_{base_name}.jpg"), 
                          dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"创建合并图像时出错: {str(e)}")
        
        return attention_outputs
                
    def process_directory(self, input_dir, output_dir, max_images=50):
        """处理目录中的所有图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_dir, file))
        
        # 限制图像数量
        if max_images > 0 and len(image_files) > max_images:
            print(f"限制处理前 {max_images} 张图像(共 {len(image_files)} 张)")
            image_files = image_files[:max_images]
        
        # 处理每张图像
        all_results = []
        for img_path in tqdm(image_files, desc="生成注意力图"):
            img_name = os.path.basename(img_path)
            img_output_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
            os.makedirs(img_output_dir, exist_ok=True)
            
            results = self.visualize_attention(img_path, img_output_dir)
            if results:
                all_results.extend(results)
                
        return all_results
        
def parse_args():
    parser = argparse.ArgumentParser(description='可视化ViT-NPR模型的注意力机制')
    parser.add_argument('--model_path', type=str, required=True, 
                      help='模型路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图像或目录路径')
    parser.add_argument('--output_dir', type=str, default='./attention_vis',
                      help='输出目录')
    parser.add_argument('--max_images', type=int, default=50,
                      help='处理的最大图像数量(用于目录)')
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # 创建注意力可视化器
    visualizer = AttentionVisualizer(args.model_path)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 处理单个文件
        print(f"处理图像: {args.input}")
        output_paths = visualizer.visualize_attention(args.input, args.output_dir)
        if output_paths:
            print(f"生成了 {len(output_paths)} 个注意力图，保存至: {args.output_dir}")
    else:
        # 处理目录
        print(f"处理目录: {args.input}")
        output_paths = visualizer.process_directory(args.input, args.output_dir, args.max_images)
        print(f"生成了 {len(output_paths)} 个注意力图，保存至: {args.output_dir}")
    
if __name__ == "__main__":
    main() 