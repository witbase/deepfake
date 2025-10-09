import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from validate import validate
from data import create_dataloader
from networks.vit_npr import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
import torch.nn.functional as F

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

# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        batch_size = features.shape[0]
        labels = torch.arange(batch_size, device=features.device)
        
        # 计算相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # 应用temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # 计算对比损失
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

class VitTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_model(num_classes=1)
        self.model = self.model.to(self.device)
        
        # 定义损失函数和优化器
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt.lr,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=opt.niter
        )
        
        # 添加对比损失
        self.contrastive_criterion = ContrastiveLoss()
        
        self.total_steps = 0
        self.lr = opt.lr
        
    def set_input(self, data):
        self.image = data[0].to(self.device)
        # 确保标签维度正确 [B] -> [B,1]
        self.label = data[1].unsqueeze(1).float().to(self.device)
        
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        
        # 前向传播(包括特征提取)
        output = self.model(self.image)
        features = self.model(self.image, return_features=True)
        
        # 计算分类损失和对比损失
        cls_loss = self.criterion(output, self.label)
        contrastive_loss = self.contrastive_criterion(features)
        
        # 总损失
        self.loss = cls_loss + 0.1 * contrastive_loss
        
        # 梯度裁剪和反向传播
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def adjust_learning_rate(self):
        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        
    def save_networks(self, epoch):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.loss,
        }, save_path)
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    
    # 设置数据路径
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    
    # 设置日志
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))
    
    # 获取验证配置
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    
    # 创建数据加载器
    data_loader = create_dataloader(opt)
    
    # 创建tensorboard写入器
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    # 创建模型
    model = VitTrainer(opt)
    
    def testmodel():
        print('*'*25)
        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = True
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc)
            aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
        print('*'*25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    
    # 开始训练
    model.train()
    print(f'cwd: {os.getcwd()}')
    
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters()
            
            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 
                      "Train loss: {} at step: {} lr {}".format(
                          model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
        
        # 调整学习率
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 
                  'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()
        
        # 验证
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        model.train()
    
    # 最终测试
    model.eval()
    testmodel()
    model.save_networks('last') 