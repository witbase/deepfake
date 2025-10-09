#!/usr/bin/env python
import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='运行深度伪造检测模型的可视化工具')
    
    # 基本参数
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图像或包含图像的目录')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                      help='存储可视化结果的目录')
    
    # 可视化类型
    parser.add_argument('--type', type=str, default='all',
                      choices=['feature', 'attention', 'all'],
                      help='可视化的类型: feature=特征图, attention=注意力图, all=两者都生成')
    
    # 其他参数
    parser.add_argument('--max_images', type=int, default=50,
                      help='如果输入是目录，最多处理的图像数量')
    parser.add_argument('--feature_mode', type=str, default='overlay',
                      choices=['overlay', 'heatmap'],
                      help='特征图的可视化模式')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件是否存在
    scripts = {
        'feature': 'visualize_features.py',
        'attention': 'attention_vis.py',
    }
    
    for key, script in scripts.items():
        if not os.path.isfile(script):
            print(f"错误: 找不到所需的脚本 {script}")
            sys.exit(1)
    
    # 检查模型文件
    if not os.path.isfile(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        sys.exit(1)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在 {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行所选的可视化脚本
    if args.type in ['feature', 'all']:
        print("\n==== 生成特征图 ====")
        feature_output = os.path.join(args.output_dir, 'feature_maps')
        cmd = [
            sys.executable, scripts['feature'],
            '--model_path', args.model_path,
            '--input', args.input,
            '--output_dir', feature_output,
            '--max_images', str(args.max_images),
            '--type', args.feature_mode
        ]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    if args.type in ['attention', 'all']:
        print("\n==== 生成注意力图 ====")
        attention_output = os.path.join(args.output_dir, 'attention_maps')
        cmd = [
            sys.executable, scripts['attention'],
            '--model_path', args.model_path,
            '--input', args.input,
            '--output_dir', attention_output,
            '--max_images', str(args.max_images)
        ]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    print(f"\n所有可视化结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 