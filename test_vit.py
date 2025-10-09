import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.vit_npr import create_model
from options.test_options import TestOptions
import numpy as np
import random

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

# 测试配置
DetectionTests = {
    'ForenSynths': { 
        'dataroot': '/app/network/ls-deepfake-detection/datasets/ForenSynths/',
        'no_resize': False,
        'no_crop': True,
    },
    'GANGen-Detection': { 
        'dataroot': '/app/network/ls-deepfake-detection/datasets/GANGen-Detection/',
        'no_resize': True,
        'no_crop': True,
    },
    'DiffusionForensics': { 
        'dataroot': '/app/network/ls-deepfake-detection/datasets/DiffusionForensics/',
        'no_resize': False,
        'no_crop': True,
    },
    'UniversalFakeDetect': { 
        'dataroot': '/app/network/ls-deepfake-detection/datasets/UniversalFakeDetect/',
        'no_resize': False,
        'no_crop': True,
    },
}

def test_model(model_path, batch_size=32):
    """测试模型在各个数据集上的性能"""
    opt = TestOptions().parse(print_options=False)
    opt.model_path = model_path
    opt.batch_size = batch_size
    print(f'Model_path: {opt.model_path}')
    print(f'Batch size: {opt.batch_size}')
    
    # 创建模型
    model = create_model(num_classes=1)
    
    # 加载模型权重
    print('Loading checkpoint...')
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print('Loading from new checkpoint format...')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
        print(f'Last loss: {checkpoint.get("loss", "unknown")}')
    else:
        print('Loading from old checkpoint format...')
        model.load_state_dict(checkpoint)
    
    model.cuda()
    model.eval()
    
    # 测试结果汇总
    all_results = {}
    
    for testSet in DetectionTests.keys():
        dataroot = DetectionTests[testSet]['dataroot']
        printSet(testSet)
        print('-' * 50)
        
        accs = []
        aps = []
        print(f'Testing started at: {time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}')
        
        # 测试每个子数据集
        for v_id, val in enumerate(sorted(os.listdir(dataroot))):
            opt.dataroot = os.path.join(dataroot, val)
            opt.classes = ''
            opt.no_resize = DetectionTests[testSet]['no_resize']
            opt.no_crop = DetectionTests[testSet]['no_crop']
            
            # 执行验证
            print(f'Testing on {val}...')
            acc, ap, _, _, _, _ = validate(model, opt)
            accs.append(acc)
            aps.append(ap)
            print(f"({v_id:2d} {val:12}) acc: {acc*100:5.1f}; ap: {ap*100:5.1f}")
        
        # 计算平均性能
        mean_acc = np.array(accs).mean() * 100
        mean_ap = np.array(aps).mean() * 100
        print(f"\nMean Performance on {testSet}:")
        print(f"Average Accuracy: {mean_acc:5.1f}")
        print(f"Average AP: {mean_ap:5.1f}")
        print('*' * 50)
        
        # 保存结果
        all_results[testSet] = {
            'accuracy': mean_acc,
            'ap': mean_ap,
            'individual_accs': accs,
            'individual_aps': aps
        }
    
    # 打印总体结果
    print("\nOverall Results Summary:")
    print('=' * 50)
    for testSet, results in all_results.items():
        print(f"{testSet:20} - Acc: {results['accuracy']:5.1f}, AP: {results['ap']:5.1f}")
    print('=' * 50)
    
    return all_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test ViT model on multiple datasets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    args = parser.parse_args()
    
    test_model(args.model_path, args.batch_size) 