import cv2
import math
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

        # ==================== 1. 基础配置与全局参数 ====================
        self.cfg = cfg
        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]
        self.window_size = cfg.window_size
        self.c_feat = 64

        # 生成基础坐标网格 (用于后续伪光流对齐)
        self.grid_generator()

        # 生成 64 维 2D 绝对位置编码 (供下游回归任务使用)
        self.pe = positionalencoding2d(d_model=self.c_feat,
                                       height=self.cfg.height // self.seg_sf[0],
                                       width=self.cfg.width // self.seg_sf[0]).cuda()

        # ==================== 2. 基础特征提取模块 ====================
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

        # 空间加权卷积 (Trick 1: 将 guide 提纯为 1 个通道的空间权重图)
        self.guide_weight_conv = nn.Conv2d(self.c_feat, 1, kernel_size=1)

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )

        # ==================== 3. 隐式形变对齐网络 (替代传统光流) ====================
        # 共享运动特征提取 (极轻量)
        self.align_feat_extractor = nn.Sequential(
            conv_bn_relu(self.c_feat * 2, self.c_feat, kernel_size=3, padding=1),
            conv_bn_relu(self.c_feat, self.c_feat, kernel_size=3, padding=1)
        )

        # Head 1: 预测 DCN 偏移量 (18通道)，专为高维特征图对齐服务
        self.offset_head = nn.Conv2d(self.c_feat, 18, kernel_size=1)

        # Head 2: 预测轻量级伪光流 (2通道)，专为生成 grid 和下游 Loss 服务
        self.flow_head = nn.Conv2d(self.c_feat, 2, kernel_size=1)

        # 特征对齐使用的 DCN (将前一帧特征 Warp 到当前帧)
        self.align_dcn = Deformable_Conv2d(in_channels=self.c_feat, out_channels=self.c_feat,
                                           kernel_size=3, stride=1, padding=1)

        # ==================== 4. 残差 Mamba 时序门控模块 ====================
        # 输入投影：当前帧(64) + 残差特征(64) = 128 -> 映射到 Mamba 需要的 192 维
        self.mamba_proj_in = nn.Conv2d(self.c_feat * 2, 192, kernel_size=1)

        # Mamba 专用的 192 维 2D 正余弦位置编码 (Trick 2)
        self.pe_192 = positionalencoding2d(d_model=192,
                                           height=self.cfg.height // self.seg_sf[0],
                                           width=self.cfg.width // self.seg_sf[0]).cuda()

        # 双向扫描 Mamba (水平 + 垂直)
        self.fusion_mamba = BiMambaBlock(d_model=192)
        self.norm = nn.LayerNorm(192)

        self.fusion_mamba_v = BiMambaBlock(d_model=192)
        self.norm_v = nn.LayerNorm(192)

        # Mamba 输出映射：降维并生成 0~1 的空间融合门控权重 (Gate)
        self.mamba_gate_out = nn.Sequential(
            nn.Conv2d(192, self.c_feat, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.c_feat),
            nn.Conv2d(self.c_feat, self.c_feat, kernel_size=1),
            nn.Sigmoid()  # 严格限制在 0~1 之间，用于时序软融合
        )

        # [初始化技巧] 让门控偏置初始为一个小负数，sigmoid(-3) ≈ 0.04
        # 确保网络初期非常保守，主要依赖当前帧，等 Mamba 学好后再动态打开门控
        nn.init.constant_(self.mamba_gate_out[-2].weight, 0)
        nn.init.constant_(self.mamba_gate_out[-2].bias, -3.0)

        # ==================== 5. 车道线曲线系数回归模块 ====================
        self.feat_embedding = torch.nn.Sequential(
            conv_bn_relu(1, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        self.regressor = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, 1, padding=2, dilation=2),
        )

        self.offset_regression = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=2, dilation=2),
            torch.nn.Conv2d(self.c_feat, 2 * 3 * 3, 1)
        )

        # 负责最终车道线参数预测的 DCN
        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_feat, out_channels=self.cfg.top_m,
                                               kernel_size=3, stride=1, padding=1)
        # 定义一个处理通道特征的 KAN 层
        self.kan_regressor = KANLinear(in_features=self.c_feat, out_features=self.c_feat)
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
        # 1. 提取基础特征
        query_feat = self.feat_embed(query_data)
        key_feat = self.feat_embed(key_data)

        b, c, h, w = query_feat.shape

        # ==================== 【极其关键的优化点】 ====================
        # 给光流匹配加上 2D 绝对坐标！
        # 这样网络在计算 correlation 的时候，如果两个像素长得一样，
        # 它会优先匹配物理距离更近、空间相对位置更合理的那一个，极大减少空旷路面的误匹配！
        pe_expanded = self.pe.expand(b, -1, h, w)
        query_feat_pe = query_feat + pe_expanded
        key_feat_pe = key_feat + pe_expanded
        # ==============================================================

        # 2. 计算代价体积 (Cost Volume) 时，使用带坐标的特征
        cost_v = self.local_window_attention(query_feat_pe, key_feat_pe)

        # 3. 预测光流时，拼接 cost_v 和 【原始特征 query_feat】
        # 注意：这里拼原始特征就行，保留最纯粹的语义信息供后续卷积网络推理
        x = torch.cat((cost_v, query_feat), dim=1)

        flow = self.flow_estimator(x)
        flow = torch.nn.functional.interpolate(flow, scale_factor=4, mode='bilinear')

        b_f = len(flow)
        grid = self.grid_xy.expand(b_f, -1, -1, -1) + flow.permute(0, 2, 3, 1)

        return flow, grid

    def forward_for_mask_generation(self, query_data, key_data):
        data_combined = torch.cat((query_data, key_data), dim=1)
        mask = self.mask_generator(data_combined)
        mask = torch.sigmoid(mask)
        return mask

    def forward_for_feat_aggregation(self, is_training=False):
        key_t_0 = f't-{0}'
        curr_refined_feat = self.memory['img_feat'][key_t_0].detach().clone().contiguous()

        # 初始化返回值变量，以防 num_t 为 0 的边界情况
        aligned_key_guide = None
        aligned_key_probmap = None
        grid_c = None

        for t in range(1, self.cfg.num_t + 1):
            key_t = f't-{t}'
            key_img_feat = self.memory['img_feat'][key_t].detach().clone().contiguous()
            key_guide = self.memory['guide_cls'][key_t].detach().clone().contiguous()
            key_probmap = self.memory['prob_map'][key_t][:, 1:].detach().clone().contiguous()

            # ========== 【步骤 1: 提取共享运动特征】 ==========
            concat_for_align = torch.cat([curr_refined_feat, key_img_feat], dim=1)
            align_feat = self.align_feat_extractor(concat_for_align)

            # ========== 【步骤 2: DCN 处理高维特征 (Head 1)】 ==========
            offset = self.offset_head(align_feat)
            aligned_key_img_feat = self.align_dcn(key_img_feat, offset)

            # ========== 【步骤 3: 伪光流生成 Grid 与辅助对齐 (Head 2)】 ==========
            # 直接预测稠密伪光流
            flow_c = self.flow_head(align_feat)
            b_f = len(flow_c)
            # 生成标准的特征级网格 (格式: [B, H, W, 2])
            grid_c = self.grid_xy.expand(b_f, -1, -1, -1) + flow_c.permute(0, 2, 3, 1)

            # 用伪光流对 downstream 需要的低维标签/掩码进行插值对齐
            aligned_key_guide = F.grid_sample(key_guide, grid_c, mode='bilinear', padding_mode='zeros')
            aligned_key_probmap = F.grid_sample(key_probmap, grid_c, mode='bilinear', padding_mode='zeros')

            # ========== 【步骤 4: 构建时序残差与 Mamba 门控】 ==========
            delta_F = curr_refined_feat - aligned_key_img_feat

            mamba_in = torch.cat([curr_refined_feat, delta_F], dim=1)
            mamba_in = self.mamba_proj_in(mamba_in)

            b, c_m, h, w = mamba_in.shape
            mamba_in = mamba_in + self.pe_192.expand(b, -1, -1, -1)

            x_h = mamba_in.flatten(2).transpose(1, 2)
            x_h = self.norm(x_h)
            x_h = checkpoint(self.fusion_mamba, x_h)
            out_h = x_h.transpose(1, 2).view(b, c_m, h, w)

            x_v = mamba_in.transpose(2, 3).flatten(2).transpose(1, 2)
            x_v = self.norm_v(x_v)
            x_v = checkpoint(self.fusion_mamba_v, x_v)
            out_v = x_v.transpose(1, 2).view(b, c_m, w, h).transpose(2, 3)

            x_mamba_out = out_h + out_v
            gate = self.mamba_gate_out(x_mamba_out)

            # 动量残差更新
            curr_refined_feat = (1.0 - gate) * curr_refined_feat + gate * aligned_key_img_feat

        self.img_feat = curr_refined_feat

        # 完美兼容你原始的接口！
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

        # ==================== 【消融实验: 暂时移除 KAN 的部分】 ====================
        # x_flat = x.flatten(2).transpose(1, 2).reshape(-1, x.size(1))
        # x_kan = self.kan_regressor(x_flat)
        # x_kan = x_kan.view(b, h * w, -1).transpose(1, 2).view(b, -1, h, w)
        # x = x + x_kan
        # =========================================================================

        # 4. 利用形变卷积输出最终的曲线系数 (直接用纯 CNN 的 x 去预测)
        coeff_map = self.deform_conv2d(x, offset)

        return {'coeff_map': coeff_map}


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
