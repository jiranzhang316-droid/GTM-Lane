import cv2
import math
import time
from collections import defaultdict
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.backbone import *
from libs.utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    基于 RBF (Radial Basis Function) 的极其稳定的 Fast-KAN 线性层。
    无任何复杂切片，完美适配 GPU，无维度报错风险。
    """

    def __init__(self, in_features, out_features, grid_size=5):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # 1. 传统的线性基础部分 (Base)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = nn.SiLU()

        # 2. RBF 网格中心和宽度 (Spline/RBF 替代)
        # 将特征空间均匀划分为 grid_size 个基函数
        grid = torch.linspace(-1.0, 1.0, steps=grid_size, dtype=torch.float32)
        self.register_buffer("grid", grid)  # shape: (grid_size,)

        # 控制高斯分布的宽度
        self.denominator = (2.0 / (grid_size - 1)) ** 2

        # 3. KAN 非线性映射权重
        # 每个输入特征都有 grid_size 个控制点，映射到 out_features
        self.kan_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming 初始化保证训练初期的稳定性
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.kan_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        # x shape: [B*H*W, in_features]

        # --- 1. 基础线性映射 ---
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # --- 2. KAN 高阶非线性映射 (RBF 展开) ---
        # 扩展 x 的维度以计算与 grid 的距离: [B*H*W, in_features, 1]
        x_expanded = x.unsqueeze(-1)

        # 计算高斯径向基激活 (无任何复杂切片，绝对不会报错)
        # rbf shape: [B*H*W, in_features, grid_size]
        rbf = torch.exp(-((x_expanded - self.grid) ** 2) / self.denominator)

        # 展平 RBF 特征: [B*H*W, in_features * grid_size]
        # 展平 RBF 特征: [B*H*W, in_features * grid_size]
        rbf_flat = rbf.reshape(x.size(0), -1)

        # 展平 KAN 权重: [out_features, in_features * grid_size]
        kan_weight_flat = self.kan_weight.view(self.out_features, -1)

        # 线性组合 RBF 激活结果
        kan_output = F.linear(rbf_flat, kan_weight_flat)

        # 最终输出 = 基础映射 + 高阶映射
        return base_output + kan_output
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.view(1, d_model * 2, height, width)
    return pe

class Deformable_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Deformable_Conv2d, self).__init__()
        self.deform_conv2d = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, offset, mask=None):
        out = self.deform_conv2d(x, offset, mask)
        return out

class Conv_ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Conv_ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += x
        out = self.relu(out)
        return out


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class conv1d_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.window_size = cfg.window_size
        self.c_feat = 64

        self.flow_estimator = torch.nn.Sequential(
            conv_relu(self.cfg.window_size**2 + self.c_feat, self.c_feat, kernel_size=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=2, padding=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=2, dilation=2),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=2, padding=1),
            conv_relu(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=2, dilation=2),
            Conv_ResBlock(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=1),
            Conv_ResBlock(self.c_feat, self.c_feat, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, kernel_size=1),
        )

        self.feat_embed = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
        )

        self.feat_guide = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, self.c_feat, 1),
        )

        self.feat_aggregator = torch.nn.Sequential(
            conv_bn_relu(self.c_feat * 3, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, self.c_feat, 1)
        )

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )
        self.grid_generator()

        self.feat_embedding = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        self.regressor = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        kernel_size = 3
        self.offset_regression = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, 2 * kernel_size * kernel_size, 1)
        )

        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_feat, out_channels=self.cfg.top_m,
                                               kernel_size=kernel_size, stride=1, padding=1)

        self.pe = positionalencoding2d(d_model=self.c_feat, height=self.cfg.height // self.seg_sf[0], width=self.cfg.width // self.seg_sf[0]).cuda()

        # ========== 【新增 Trick 1: 空间加权卷积】 ==========
        # 用 1x1 卷积将 guide 提纯为 1 个通道的空间权重图
        self.guide_weight_conv = nn.Conv2d(self.c_feat, 1, kernel_size=1)

        # ========== 【新增 Trick 2: 给 Mamba 专用的 192 维 2D 位置编码】 ==========
        # 直接生成 [1, 192, H, W] 的 2D 正余弦位置编码，完美保留绝对坐标
        self.pe_192 = positionalencoding2d(d_model=192, height=self.cfg.height // self.seg_sf[0],
                                           width=self.cfg.width // self.seg_sf[0]).cuda()

        self.fusion_mamba = BiMambaBlock(d_model=192)
        self.norm = nn.LayerNorm(192)
        self.final_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 2. 垂直方向 Mamba (新增的！！！)
        # 目的：专门捕捉车道线的纵向连续性，解决 mIoU 低的问题
        self.fusion_mamba_v = BiMambaBlock(d_model=192)
        self.norm_v = nn.LayerNorm(192)
        # 在初始化中添加
        self.gate = nn.Parameter(torch.zeros(1))  # 初始化为0，让模型从原始特征开始学
        # 3. 最后的降维映射
        self.final_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 4. 空间细化模块 (Spatial Refinement)
        # 你的定义已经是正确的了 (64*2 -> 64 -> 1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # 输出通道必须为 1，用于生成权重图
        )
        # [新增初始化] 让门控偏置初始为一个小负数，sigmoid(-3) ≈ 0.04
        # 这样网络一开始会非常保守，几乎只用原始特征，等 Mamba 慢慢学好后再打开门控
        nn.init.constant_(self.spatial_attn[-1].weight, 0)
        nn.init.constant_(self.spatial_attn[-1].bias, -3.0)

        # 定义一个处理通道特征的 KAN 层
        self.kan_regressor = KANLinear(in_features=self.c_feat, out_features=self.c_feat)

        # ========== 【耗时统计】 ==========
        self.enable_timing = False  # 默认关闭，测试时由外部开启
        self.reset_timing()
    def grid_generator(self):
        x = np.linspace(0, self.cfg.width // self.seg_sf[0] - 1, self.cfg.width // self.seg_sf[0])
        y = np.linspace(0, self.cfg.height // self.seg_sf[0] - 1, self.cfg.height // self.seg_sf[0])
        grid_xy = np.float32(np.meshgrid(x, y))
        _, h, w = grid_xy.shape
        self.grid_xy = to_tensor(grid_xy).permute(1, 2, 0).view(1, h, w, 2)
        self.grid_xy[:, :, :, 0] = (self.grid_xy[:, :, :, 0] / (self.cfg.width // self.seg_sf[0] - 1) - 0.5) * 2
        self.grid_xy[:, :, :, 1] = (self.grid_xy[:, :, :, 1] / (self.cfg.height // self.seg_sf[0] - 1) - 0.5) * 2

    def local_window_attention(self, query_data, key_data):
        b, c, h, w = key_data.shape
        query_data = query_data.permute(0, 2, 3, 1).reshape(-1, c, 1)
        key_data = F.unfold(key_data, kernel_size=(self.window_size, self.window_size), stride=1, padding=(self.window_size // 2, self.window_size // 2)).view(b, c, self.window_size**2, h, w).permute(0, 3, 4, 1, 2)
        key_data = key_data.reshape(-1, c, self.window_size**2)

        correlation = torch.bmm(query_data.permute(0, 2, 1), key_data) / (c ** 0.5)
        sim_map = F.softmax(correlation, dim=2)
        sim_map = sim_map.view(b, h, w, self.window_size**2).permute(0, 3, 1, 2)
        return sim_map

    def forward_for_flow_estimation(self, query_data, key_data):
        # ========== 【计时：光流对齐】 ==========
        if self.enable_timing:
            torch.cuda.synchronize()
            _t_start = time.perf_counter()

        # 1. 提取基础特征
        query_feat = self.feat_embed(query_data)
        key_feat = self.feat_embed(key_data)

        b, c, h, w = query_feat.shape

        # ==================== 【极其关键的优化点】 ====================
        pe_expanded = self.pe.expand(b, -1, h, w)
        query_feat_pe = query_feat + pe_expanded
        key_feat_pe = key_feat + pe_expanded
        # ==============================================================

        # 2. 计算代价体积 (Cost Volume) 时，使用带坐标的特征
        cost_v = self.local_window_attention(query_feat_pe, key_feat_pe)

        # 3. 预测光流时，拼接 cost_v 和 【原始特征 query_feat】
        x = torch.cat((cost_v, query_feat), dim=1)

        flow = self.flow_estimator(x)
        flow = torch.nn.functional.interpolate(flow, scale_factor=4, mode='bilinear')

        b_f = len(flow)
        grid = self.grid_xy.expand(b_f, -1, -1, -1) + flow.permute(0, 2, 3, 1)

        # ========== 【计时：光流对齐结束】 ==========
        if self.enable_timing:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - _t_start
            self.timing['flow_alignment'].append(elapsed * 1000)

        return flow, grid

    def forward_for_mask_generation(self, query_data, key_data):
        data_combined = torch.cat((query_data, key_data), dim=1)
        mask = self.mask_generator(data_combined)
        mask = torch.sigmoid(mask)
        return mask

    def forward_for_feat_aggregation(self, is_training=False):
        # T=0 (当前帧) 原始特征，作为最清晰的 Anchor
        key_t_0 = f't-{0}'
        query_img_feat_orig = self.memory['img_feat'][key_t_0].detach().clone().contiguous()

        # 用于迭代更新的特征，初始状态就是当前帧
        curr_refined_feat = query_img_feat_orig

        aligned_key_guide = None
        aligned_key_probmap = None
        grid_c = None

        # -------------------------------------------------------------------------
        # 注意：为了支持下面的双向扫描，请确保你在 __init__ 中定义了以下新层：
        # self.fusion_mamba_v = ... (同 fusion_mamba配置)
        # self.norm_v = nn.LayerNorm(...)
        # self.pos_embed_v = ... (形状要适配 W*H)
        # 如果暂时没定义，可以先注释掉垂直分支，只用水平分支，但效果会打折。
        # -------------------------------------------------------------------------

        for t in range(1, self.cfg.num_t + 1):
            key_t = f't-{t}'
            key_img_feat = self.memory['img_feat'][key_t].detach().clone().contiguous()
            key_guide = self.memory['guide_cls'][key_t].detach().clone().contiguous()
            key_probmap = self.memory['prob_map'][key_t][:, 1:].detach().clone().contiguous()

            # 1. 光流对齐 (这一步必须保留，给 Mamba 提供对齐的基础)
            # 使用最原始的 query 进行光流估计，保证稳定性
            flow_c, grid_c = self.forward_for_flow_estimation(query_img_feat_orig, key_img_feat)

            aligned_key_img_feat = F.grid_sample(key_img_feat, grid_c, mode='bilinear', padding_mode='zeros')
            aligned_key_guide = F.grid_sample(key_guide, grid_c, mode='bilinear', padding_mode='zeros')
            aligned_key_probmap = F.grid_sample(key_probmap, grid_c, mode='bilinear', padding_mode='zeros')

            feat_guide = self.forward_for_guidance_feat(aligned_key_guide)
            # 2. 极简特征拼接：保留最原始、最丰富的上下文信息
            # 让后续强大的 Mamba 模块自己去学习如何利用 feat_guide 进行注意力选择
            combined = torch.cat((curr_refined_feat, aligned_key_img_feat, feat_guide), dim=1)
            b, c, h, w = combined.shape
            # ========== 【应用 Trick 2: 注入 2D 绝对位置编码】 ==========
            # 在被展平送入 Mamba 之前，先让每个像素知道自己在画面中的绝对 (x, y) 坐标
            combined = combined + self.pe_192.expand(b, -1, -1, -1)

            # === 3. 双向 Mamba 扫描 ===

            # ========== 【计时：BiMamba 聚合】 ==========
            if self.enable_timing:
                torch.cuda.synchronize()
                _t_mamba = time.perf_counter()

            # [分支 A: 水平扫描]
            x_h = combined.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_h = self.norm(x_h)
            x_h = checkpoint(self.fusion_mamba, x_h)
            out_h = x_h.transpose(1, 2).view(b, c, h, w)

            # [分支 B: 垂直扫描]
            x_v = combined.transpose(2, 3).flatten(2).transpose(1, 2)  # [B, W*H, C]
            x_v = self.norm_v(x_v)
            x_v = checkpoint(self.fusion_mamba_v, x_v)
            out_v = x_v.transpose(1, 2).view(b, c, w, h).transpose(2, 3)

            # 融合双向特征
            x_mamba_out = out_h + out_v

            # 降维/映射回原始通道
            x_refined_candidate = self.final_conv(x_mamba_out)

            # ========== 【计时：BiMamba 聚合结束】 ==========
            if self.enable_timing:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - _t_mamba
                self.timing['bimamba_aggregation'].append(elapsed * 1000)

            # === 4. 空间动态门控 (Spatial Dynamic Gating) ===
            # 目的：解决 mIoU 低的问题。
            # 逻辑：计算一张权重图 Alpha。
            # Alpha = 1: 相信 Mamba (用于填补遮挡、断裂)
            # Alpha = 0: 相信原始 Query (用于保持清晰边缘)
            # 输入：Mamba 修正后的特征 + 原始最清晰的 Query
            gate_input = torch.cat([x_refined_candidate, curr_refined_feat], dim=1)
            spatial_gate = torch.sigmoid(self.spatial_attn(gate_input))
            # 1. 计算光流场的模长 (运动剧烈程度)
            # flow_c shape: [B, 2, H, W]
            flow_magnitude = torch.norm(flow_c, p=2, dim=1, keepdim=True)

            # 2. 将光流异常转化为 0~1 的置信度衰减系数 (可学习温度系数或固定经验值)
            # 当 flow_magnitude 很大 (比如遇到了前方卡车/动态遮挡)，衰减系数趋近于 0
            tau = 5.0  # 温度超参数，可根据 VIL-100/OpenLane-V 数据集微调
            flow_confidence = torch.exp(-flow_magnitude / tau)

            # 3. 终极门控 = 语义门控 * 光流置信度
            # 如果没遮挡(flow平滑)，由 spatial_gate 主导；如果遮挡(flow剧变)，强制掐断更新！
            final_gate = spatial_gate * flow_confidence
            # 迭代更新：被遮挡区域 final_gate 接近 0，完美保留历史干净特征
            curr_refined_feat = curr_refined_feat * (1.0 - final_gate) + x_refined_candidate * final_gate
        # 循环结束，赋值给 self.img_feat
        self.img_feat = curr_refined_feat

        return {
            'key_probmap': self.memory['prob_map'][f't-{self.cfg.num_t}'][:, 1:],
            'key_guide': self.memory['guide_cls'][f't-{self.cfg.num_t}'],
            'aligned_key_probmap': aligned_key_probmap,
            'aligned_key_guide': aligned_key_guide,
            'grid': grid_c
        }

    def forward_for_guidance_feat(self, guide_map):
        # data = torch.cat((prob_map, guide_map), dim=1)
        feat_guide = self.feat_guide(guide_map)
        return feat_guide

    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        self.prob_map = F.softmax(out, dim=1)
        return {'seg_map_logit': out,
                'seg_map': self.prob_map[:, 1:2]}

    def forward_for_regression(self):
        b, c, h, w = self.prob_map.shape

        # 1. 基础特征提取
        feat_c = self.feat_embedding(self.prob_map[:, 1:].detach())
        feat_c = feat_c + self.pe.expand(b, -1, -1, -1)

        # 2. 预测形变偏移量 (保持原有 CNN 即可，因为依赖空间局部性)
        offset = self.offset_regression(feat_c)

        # 3. 基础回归特征
        x = self.regressor(feat_c)  # shape: [B, C, H, W]

        # ==================== 【融入 KAN 的部分】 ====================
        # 因为 KAN 擅长处理 1D 向量的非线性映射，我们把每个像素的特征抽出来给 KAN

        # ========== 【计时：Fast-KAN 解码】 ==========
        if self.enable_timing:
            torch.cuda.synchronize()
            _t_kan = time.perf_counter()

        # 展平: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        x_flat = x.flatten(2).transpose(1, 2).reshape(-1, x.size(1))

        # 通过 KAN 进行极高阶的非线性特征映射（学习复杂的系数组合）
        x_kan = self.kan_regressor(x_flat)

        # 还原回 2D 形状: [B*H*W, C] -> [B, H*W, C] -> [B, C, H, W]
        x_kan = x_kan.view(b, h * w, -1).transpose(1, 2).view(b, -1, h, w)

        # (可选) 残差连接，防止 KAN 初始化不好导致训练崩溃
        x = x + x_kan
        # =============================================================

        # ========== 【计时：Fast-KAN 解码结束】 ==========
        if self.enable_timing:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - _t_kan
            self.timing['kan_regression'].append(elapsed * 1000)

        # 4. 利用形变卷积输出最终的曲线系数
        coeff_map = self.deform_conv2d(x, offset)

        return {'coeff_map': coeff_map}

    def reset_timing(self):
        """重置所有计时统计"""
        self.timing = defaultdict(list)

    def get_timing_report(self):
        """
        打印详细的耗时统计报告
        每帧统计：光流对齐 / BiMamba递归聚合 / Fast-KAN解码
        """
        stages = [
            ('flow_alignment',       '光流对齐 (Flow Alignment)       '),
            ('bimamba_aggregation',  'BiMamba 递归聚合 (BiMamba Aggr)'),
            ('kan_regression',       'Fast-KAN 解码 (KAN Regression)  '),
        ]

        if len(self.timing.get('flow_alignment', [])) == 0:
            print("\n[计时统计] 没有收集到耗时数据，请先设置 model.enable_timing = True\n")
            return None

        print("\n" + "=" * 70)
        print("  RPLD 模型各阶段耗时统计 (GPU Wall Time)")
        print("=" * 70)

        total_sum = 0.0
        for key, name in stages:
            times = self.timing.get(key, [])
            if len(times) == 0:
                print(f"  {name}: 无数据")
                continue

            avg = np.mean(times)
            std = np.std(times)
            med = np.median(times)
            mn = np.min(times)
            mx = np.max(times)
            total = np.sum(times)
            total_sum += total

            print(f"  {name}")
            print(f"    调用次数: {len(times):>6d}  总耗时: {total:>8.2f} ms")
            print(f"    均值: {avg:>8.2f} ms  标准差: {std:>8.2f} ms")
            print(f"    中位数: {med:>8.2f} ms  最小值: {mn:>8.2f} ms  最大值: {mx:>8.2f} ms")
            print()

        if total_sum > 0:
            print(f"  {'三阶段合计':<30s}")
            print(f"    总耗时: {total_sum:>8.2f} ms")
            print()
            print(f"  {'阶段耗时占比':<30s}")
            for key, name in stages:
                times = self.timing.get(key, [])
                if len(times) > 0:
                    pct = np.sum(times) / total_sum * 100
                    bar_len = int(pct / 2)
                    bar = '|' * bar_len + '.' * (50 - bar_len)
                    print(f"    {name}: {bar} {pct:5.1f}%")

        print("=" * 70 + "\n")

        return {
            key: {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'median': float(np.median(v)),
                'min': float(np.min(v)),
                'max': float(np.max(v)),
                'count': len(v),
                'total_ms': float(np.sum(v)),
            }
            for key, v in self.timing.items() if len(v) > 0
        }


class BiMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # 1. 投影层
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # 2. 改进一：加入 Mamba 标配的局部 1D 深度可分离卷积 (极其关键)
        # 作用：在做全局扫描前，先让相邻的像素交换一下信息，极大提升分割边缘的精度
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model  # 深度可分离，计算量极小
        )

        # 3. SSM 参数生成
        self.x_proj = nn.Linear(d_model, 16 + d_model * 2)
        self.dt_proj = nn.Linear(16, d_model)

        # 4. 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward_scan(self, x):
        """改进后的纯 PyTorch 伪扫描机制 (修复数值爆炸)"""
        # x: [B, L, d_model]
        sbc = self.x_proj(x)
        dt, B, C = torch.split(sbc, [16, self.d_model, self.d_model], dim=-1)

        # 修复 1：用 sigmoid 把更新率严格限制在 0~1 之间，防止数值失控
        dt = torch.sigmoid(self.dt_proj(dt))

        # 结合输入
        x_gated = torch.tanh(dt * x + B)

        # 🚨 修复 2：绝对不能只用 cumsum！必须计算“累积平均”！
        x_cumsum = torch.cumsum(x_gated, dim=1)
        # 生成一个 [1, L, 1] 的递增序列：1, 2, 3... L
        seq_len = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
        # 除以走过的步数，这样数值永远稳定在合理范围内！
        x_norm = x_cumsum / seq_len

        # 结合当前输出矩阵 C
        return x_norm * torch.sigmoid(C)

    def forward(self, x):
        # x 维度: [B, L, 192]
        # 改进三：保存输入，用于最后的残差连接
        residual = x

        # 1. 进入投影层并拆分
        projected = self.in_proj(x)
        res, z = projected.chunk(2, dim=-1)

        # 2. 经过 1D 局部卷积 (针对扫描分支 res)
        # Conv1d 需要输入维度为 [B, C, L]，所以需要 transpose
        res = res.transpose(1, 2)
        res = self.conv1d(res)
        res = res.transpose(1, 2)

        # 3. 激活
        res = F.silu(res)
        z = F.silu(z)

        # 4. 双向扫描 (基于序列传递)
        x_fwd = self.forward_scan(res)
        x_bwd = self.forward_scan(res.flip(dims=[1])).flip(dims=[1])

        # 5. 门控融合
        out = (x_fwd + x_bwd) * z

        # 6. 映射回原始空间，并加上残差！
        out = self.out_proj(out)
        return out + residual

# class SimpleMambaBlock(nn.Module):
#     """
#     一个无需特殊编译环境的轻量化 Mamba 风格聚合模块 (Pure PyTorch)
#     适用于 2025-2026 年视频特征融合任务
#     """
#
#     def __init__(self, d_model):
#         super().__init__()
#         self.d_model = d_model
#
#         # 1. 投影层
#         self.in_proj = nn.Linear(d_model, d_model * 2)
#
#         # 2. 核心：选择性参数生成 (Selection Mechanism)
#         # 这就是 Mamba 的灵魂：根据输入动态生成权重
#         self.x_proj = nn.Linear(d_model, 16 + d_model * 2)  # dt, B, C
#         self.dt_proj = nn.Linear(16, d_model)
#
#         # 3. 输出投影
#         self.out_proj = nn.Linear(d_model, d_model)
#
#         # 4. 辅助的局部卷积（增强局部特征）
#         self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         # x 形状: [B, L, D] (Batch, Sequence_Length, Dimension)
#         b, l, d = x.shape
#
#         # 预处理
#         x_copy = x
#         x_in = self.in_proj(x)
#         x_split, z = x_in.chunk(2, dim=-1)  # 类似于门控机制
#
#         # 局部 1D 卷积增强
#         x_split = x_split.transpose(1, 2)
#         x_split = F.silu(self.conv1d(x_split))
#         x_split = x_split.transpose(1, 2)
#
#         # --- 简化版的选择性扫描 (Selective Scan) ---
#         # 动态生成 dt, B, C
#         sbc = self.x_proj(x_split)
#         dt, B, C = torch.split(sbc, [16, d, d], dim=-1)
#         dt = F.softplus(self.dt_proj(dt))  # 步长因子
#
#         # 模拟 SSM 更新: y = Ax + Bu (这里简化为一种高效的选择性门控)
#         # 这种写法在效果上非常接近原版 Mamba，但不需要 CUDA 编译
#         strategy = torch.sigmoid(dt * x_split + B)
#         x_refined = strategy * F.silu(z)
#
#         # 最后输出
#         out = self.out_proj(x_refined)
#         return out + x_copy  # 残差连接
