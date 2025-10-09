import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import clip


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        
    def forward(self, x):
        return self.gamma * x


class NPRExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 0.5
        # 使用ResNet验证过的固定权重
        self.weight = 2.0/3.0
        
    def forward(self, x):
        # 使用nearest插值，保持与ResNet一致
        down = F.interpolate(x, scale_factor=self.factor, 
                           mode='nearest', recompute_scale_factor=True)
        up = F.interpolate(down, scale_factor=1/self.factor, 
                          mode='nearest', recompute_scale_factor=True)
        npr = x - up
        return npr * self.weight


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # 简化注意力机制，减少参数量
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 直接在序列上进行注意力计算，不需要重塑为图像
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # B, N, head, C//head
        
        # 转置为注意力计算所需的形状
        q = q.transpose(1, 2)  # B, head, N, C//head
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=7  # 这个参数在新的实现中不会被使用
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.layer_scale = LayerScale(dim)
        
    def forward(self, x):
        x = x + self.layer_scale(self.attn(self.norm1(x)))
        x = x + self.layer_scale(self.mlp(self.norm2(x)))
        return x


class TemporalModule(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        # 保持序列维度
        x, _ = self.gru(x)
        # 不再只取最后一个时间步
        return x


class NPR_CLIP_ViT(nn.Module):
    def __init__(self, clip_model="ViT-L/14", num_classes=1, 
                 patch_sizes=[16, 32], dim=768, depth=8, heads=12):
        super().__init__()
        
        # NPR特征提取器
        self.npr_extract = NPRExtractor()
        
        # 加载预训练CLIP视觉编码器
        self.clip_model, _ = clip.load(clip_model, device="cpu")
        self.visual = self.clip_model.visual
        
        # 冻结CLIP参数，只微调最后几层
        for param in self.visual.parameters():
            param.requires_grad = False
        # 微调最后4层，增加了微调层数以适应更深的模型
        for param in self.visual.transformer.resblocks[-4:].parameters():
            param.requires_grad = True
            
        # NPR特征处理
        self.npr_embeddings = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                         p1=p_size, p2=p_size),
                nn.Linear(3 * p_size * p_size, dim),
                nn.LayerNorm(dim)
            ) for p_size in patch_sizes
        ])
        
        # 计算patch数量
        self.num_patches = [(256 // p_size) ** 2 for p_size in patch_sizes]
        total_patches = sum(self.num_patches)
        
        # 位置编码和CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, total_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)
        
        # CLIP特征投影 - 调整为处理ViT-L/14的1024维特征
        clip_dim = self.visual.transformer.width  # ViT-L/14通常是1024
        self.clip_proj = nn.Linear(clip_dim, dim)
        
        # 特征融合Transformer - 增加了深度
        self.fusion_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=heads, dim_head=dim//heads, 
                                mlp_dim=dim*4, dropout=0.1)
            for _ in range(depth//2)
        ])
        
        # 自适应融合模块
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        
        # 检测头
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim//2, num_classes)
        )
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 添加层注意力机制
        self.layer_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def extract_clip_features(self, x):
        # 调整输入尺寸以匹配CLIP要求 - ViT-L/14需要224x224输入
        if x.shape[-1] != 224:
            x_clip = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        else:
            x_clip = x
            
        # 提取多层特征
        features = []
        
        # 获取patch embeddings
        x = self.visual.conv1(x_clip)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + 
                      torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        # 通过transformer块并收集特征 - 增加到收集最后4层特征
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if i >= len(self.visual.transformer.resblocks) - 4:  # 收集最后4层
                features.append(x.permute(1, 0, 2))  # LND -> NLD
        
        # 投影到相同维度
        projected_features = []
        for feat in features:
            projected_features.append(self.clip_proj(feat))
            
        return projected_features
        
    def process_npr_features(self, x):
        # NPR特征提取
        npr_feat = self.npr_extract(x)
        
        # 多尺度Patch Embedding
        embeddings = []
        for embed, num_patch in zip(self.npr_embeddings, self.num_patches):
            patch_embed = embed(npr_feat)
            embeddings.append(patch_embed)
        x = torch.cat(embeddings, dim=1)
        
        # 添加CLS token和位置编码
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        return x
    
    def adaptive_feature_fusion(self, npr_feat, clip_features):
        """
        优化的特征融合方法：
        - 使用CLIP的1、2、3层（排除第0层）
        - 高频伪影检测主要依赖NPR特征
        - Diffusion检测依赖CLIP特征
        """
        # 只选择CLIP的1、2、3层特征，排除第0层
        selected_features = clip_features[1:] if len(clip_features) > 1 else clip_features
        
        fused_features = []
        for i, clip_feat in enumerate(selected_features):
            # 调整序列长度匹配
            if clip_feat.size(1) != npr_feat.size(1):
                b, clip_seq_len, c = clip_feat.shape
                b, npr_seq_len, c = npr_feat.shape
                
                if clip_seq_len < npr_seq_len:
                    # 如果CLIP特征序列较短，使用插值扩展
                    clip_feat_reshaped = clip_feat.transpose(1, 2)
                    clip_feat_resized = F.interpolate(
                        clip_feat_reshaped, 
                        size=npr_seq_len,
                        mode='linear',
                        align_corners=False
                    )
                    clip_feat = clip_feat_resized.transpose(1, 2)
                else:
                    # 如果CLIP特征序列较长，只取前npr_seq_len个token
                    clip_feat = clip_feat[:, :npr_seq_len, :]
            
            # 计算自适应融合权重
            concat_feat = torch.cat([npr_feat, clip_feat], dim=-1)
            base_weight = self.adaptive_fusion(concat_feat)
            
            # 根据层特性调整权重
            if i == 0:  # CLIP第1层 - 更偏向NPR特征（适合高频伪影检测）
                fusion_weight = torch.clamp(base_weight + 0.3, 0.0, 1.0)
            elif i == len(selected_features) - 1:  # 最深层 - 更偏向CLIP特征（适合Diffusion检测）
                fusion_weight = torch.clamp(base_weight - 0.2, 0.0, 1.0)
            else:  # 中间层 - 平衡权重
                fusion_weight = base_weight
            
            # 加权融合
            fused = npr_feat * fusion_weight  + clip_feat *  (1 - fusion_weight)
            fused_features.append(fused)
        
        # 多尺度特征融合
        if len(fused_features) > 1:
            # 使用简单平均融合不同层的特征，避免维度问题
            final_fused = torch.mean(torch.stack(fused_features, dim=0), dim=0)
        else:
            final_fused = fused_features[0]
        
        return final_fused
        
    def forward(self, x, return_features=False):
        # 处理NPR特征
        npr_features = self.process_npr_features(x)
        
        # 提取CLIP特征
        clip_features = self.extract_clip_features(x)
        
        # 特征融合
        fused = npr_features
        
        # 注意力融合CLIP特征
        for i, block in enumerate(self.fusion_blocks):
            fused = block(fused)
            
            # 在中间层融合CLIP特征
            if i == len(self.fusion_blocks) // 2:
                # 使用改进的自适应融合方法，传入所有CLIP特征
                fused = self.adaptive_feature_fusion(fused, clip_features)
        
        # 标准化
        fused = self.norm(fused)
        
        # 取[CLS]令牌特征
        cls_feature = fused[:, 0]
        
        # 返回特征或分类结果
        if return_features:
            return self.projection_head(cls_feature)
        
        return self.mlp_head(cls_feature)


def create_npr_clip_vit(num_classes=1, pretrained=True, **kwargs):
    model = NPR_CLIP_ViT(
        clip_model="ViT-L/14",  # 升级为ViT-L/14
        num_classes=num_classes,
        patch_sizes=[16, 32],
        dim=768,  # 增加内部特征维度
        depth=8,   # 增加Transformer深度
        heads=12,  # 增加注意力头数
        **kwargs
    )
    return model
# 保持与原有代码的兼容性
def create_model(num_classes=1, pretrained=False, **kwargs):
    """
    兼容原有代码的模型创建函数，现在返回NPR_CLIP_ViT模型
    """
    return create_npr_clip_vit(num_classes=num_classes, pretrained=pretrained, **kwargs) 
