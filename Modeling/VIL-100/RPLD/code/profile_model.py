"""
RPLD 模型参数量 & FLOPs 分析脚本
使用方法: python profile_model.py
依赖: pip install thop
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np

from options.config import Config
from models.model_s import Model as Model_S
from models.model_c import Model as Model_C
from models.model import Model as Model_Main


# ============================================================
# 包装类：将多步 forward 组合为单次调用（供 thop 分析用）
# ============================================================

class ModelS_Wrapper(nn.Module):
    """model_s: 编码 → 挤压 → 分类，一步完成"""
    def __init__(self, model_s):
        super().__init__()
        self.model_s = model_s

    def forward(self, img):
        self.model_s.forward_for_encoding(img)
        self.model_s.forward_for_squeeze()
        out = self.model_s.forward_for_classification()
        return out['seg_map_logit_init']


class ModelC_Wrapper(nn.Module):
    """model_c: 概率图 → 系数图"""
    def __init__(self, model_c):
        super().__init__()
        self.model_c = model_c

    def forward(self, prob_map_1ch):
        self.model_c.prob_map = prob_map_1ch
        out = self.model_c.forward_for_regression()
        return out['coeff_map_init']


class ModelMain_FlowWrapper(nn.Module):
    """主模型：光流估计部分"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_feat, key_feat):
        flow, grid = self.model.forward_for_flow_estimation(query_feat, key_feat)
        return flow


class ModelMain_ClassifierWrapper(nn.Module):
    """主模型：分类头"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img_feat):
        self.model.img_feat = img_feat
        out = self.model.forward_for_classification()
        return out['seg_map_logit']


class ModelMain_RegressionWrapper(nn.Module):
    """主模型：回归头（含 KAN）"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prob_map):
        self.model.prob_map = prob_map
        out = self.model.forward_for_regression()
        return out['coeff_map']


# ============================================================
# 计数函数
# ============================================================

def count_params(model, name="Model"):
    """统计参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name}:")
    print(f"    总参数:    {total:>12,}  ({total/1e6:.2f} M)")
    print(f"    可训练:    {trainable:>12,}  ({trainable/1e6:.2f} M)")
    print()
    return total, trainable


def count_flops(model, inputs, name="Model"):
    """统计 FLOPs"""
    try:
        from thop import profile
        flops, params = profile(model, inputs=inputs, verbose=False)
        print(f"  {name}:")
        print(f"    FLOPs:     {flops:>12,}  ({flops/1e9:.2f} GFLOPs)")
        print(f"    Params:    {params:>12,}  ({params/1e6:.2f} M)")
        print()
        return flops, params
    except ImportError:
        print(f"  [!] 未安装 thop，跳过 FLOPs 计算。请运行: pip install thop")
        print()
        return 0, 0


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("  RPLD 模型参数量 & FLOPs 分析")
    print("=" * 60)
    print()

    # ---- 配置 ----
    cfg = Config()
    cfg.backbone = '18'
    cfg.num_t = 1
    cfg.window_size = 7

    H_img = ((cfg.height - cfg.crop_size) // 32) * 32   # 对齐到32的倍数: 256
    W_img = (cfg.width // 32) * 32                       # 640 → 640
    H_feat = cfg.height // cfg.scale_factor['seg'][0]  # 96
    W_feat = cfg.width // cfg.scale_factor['seg'][0]   # 160

    # ---- 建立模型 ----
    print(">>> 建立模型...")
    model_s = Model_S(cfg=cfg).cuda()
    model_c = Model_C(cfg=cfg).cuda()
    model_main = Model_Main(cfg=cfg).cuda()

    # ---- 1. 参数量统计 ----
    print("\n" + "-" * 40)
    print("【1. 参数量统计】\n")

    total_params = 0

    p, _ = count_params(model_s, "model_s (ResNet18 + FeatSqueeze + Classifier)")
    total_params += p

    p, _ = count_params(model_c, "model_c (FeatEmbed + DeformConv Regressor)")
    total_params += p

    # model_main 拆分各组件
    p_cls, _ = count_params(model_main.classifier, "  - classifier (2分类头)")
    total_params += p_cls

    p_flow, _ = count_params(model_main.flow_estimator, "  - flow_estimator (光流预测器)")
    total_params += p_flow
    p_flow += sum(p.numel() for p in model_main.feat_embed.parameters())

    p_feat_guide, _ = count_params(model_main.feat_guide, "  - feat_guide (引导特征编码)")
    total_params += p_feat_guide

    p_feat_agg, _ = count_params(model_main.feat_aggregator, "  - feat_aggregator (特征聚合器)")
    total_params += p_feat_agg

    p_fusion, _ = count_params(model_main.fusion_mamba, "  - fusion_mamba (水平 BiMamba)")
    total_params += p_fusion

    p_fusion_v, _ = count_params(model_main.fusion_mamba_v, "  - fusion_mamba_v (垂直 BiMamba)")
    total_params += p_fusion_v

    p_spatial, _ = count_params(model_main.spatial_attn, "  - spatial_attn (空间门控)")
    total_params += p_spatial

    p_reg, _ = count_params(model_main.regressor, "  - regressor (回归器)")
    total_params += p_reg
    p_reg2, _ = count_params(model_main.offset_regression, "  - offset_regression (偏移预测)")
    total_params += p_reg2
    p_reg3, _ = count_params(model_main.deform_conv2d, "  - deform_conv2d (可变形卷积)")
    total_params += p_reg3
    p_reg4, _ = count_params(model_main.feat_embedding, "  - feat_embedding (回归特征嵌入)")
    total_params += p_reg4

    p_kan, _ = count_params(model_main.kan_regressor, "  - kan_regressor (Fast-KAN)")
    total_params += p_kan

    p_drop, _ = count_params(model_main.final_conv, "  - final_conv (降维映射)")
    total_params += p_drop

    print(f"  {'─' * 35}")
    print(f"  【RPLD 总参数量】: {total_params:>12,}  ({total_params/1e6:.2f} M)")
    print()

    # ---- 2. FLOPs 统计 ----
    print("-" * 40)
    print("【2. FLOPs 统计 (输入 batch_size=1)】\n")

    total_flops = 0

    # 2a. model_s: 图片 → 分割图
    dummy_img = torch.randn(1, 3, H_img, W_img).cuda()
    wrapper_s = ModelS_Wrapper(model_s)
    flops, _ = count_flops(wrapper_s, (dummy_img,), f"model_s (图片→分割图) 输入: (1,3,{H_img},{W_img})")
    total_flops += flops

    # 2b. model_c: 分割图 → 系数图
    dummy_prob = torch.randn(1, 1, H_feat, W_feat).cuda()
    wrapper_c = ModelC_Wrapper(model_c)
    flops, _ = count_flops(wrapper_c, (dummy_prob,), "model_c (分割图→系数图) 输入: (1,1,96,160)")
    total_flops += flops

    # 2c. 主模型 - 光流估计
    dummy_query_feat = torch.randn(1, 64, H_feat, W_feat).cuda()
    dummy_key_feat = torch.randn(1, 64, H_feat, W_feat).cuda()
    wrapper_flow = ModelMain_FlowWrapper(model_main)
    flops, _ = count_flops(wrapper_flow, (dummy_query_feat, dummy_key_feat),
                           "主模型-光流对齐 输入: (1,64,96,160)x2")
    total_flops += flops

    # 2d. 主模型 - BiMamba 聚合 (手动估算，thop 不支持 checkpoint)
    print("  主模型-BiMamba 聚合 (含水平+垂直扫描+final_conv):")
    # BiMamba 输入是 [B, H*W, 192] 的序列
    dummy_seq = torch.randn(1, H_feat * W_feat, 192).cuda()
    flops_h, _ = count_flops(model_main.fusion_mamba, (dummy_seq,),
                             "    水平 BiMamba 输入: (1, 15360, 192)")
    flops_v, _ = count_flops(model_main.fusion_mamba_v, (dummy_seq,),
                             "    垂直 BiMamba 输入: (1, 15360, 192)")
    # final_conv
    dummy_combined = torch.randn(1, 192, H_feat, W_feat).cuda()
    flops_fc, _ = count_flops(model_main.final_conv, (dummy_combined,),
                              "    final_conv 输入: (1,192,96,160)")
    total_flops += flops_h + flops_v + flops_fc

    # 2e. 主模型 - 分类头
    dummy_img_feat = torch.randn(1, 64, H_feat, W_feat).cuda()
    wrapper_cls = ModelMain_ClassifierWrapper(model_main)
    flops, _ = count_flops(wrapper_cls, (dummy_img_feat,),
                           "主模型-分类头 输入: (1,64,96,160)")
    total_flops += flops

    # 2f. 主模型 - 回归头（含 KAN）
    dummy_prob_2ch = torch.randn(1, 2, H_feat, W_feat).cuda()
    wrapper_reg = ModelMain_RegressionWrapper(model_main)
    flops, _ = count_flops(wrapper_reg, (dummy_prob_2ch,),
                           "主模型-回归头(KAN) 输入: (1,2,96,160)")
    total_flops += flops

    # 2g. 门控部分 (spatial_attn)
    dummy_gate_input = torch.randn(1, 128, H_feat, W_feat).cuda()
    flops, _ = count_flops(model_main.spatial_attn, (dummy_gate_input,),
                           "主模型-空间门控 输入: (1,128,96,160)")
    total_flops += flops

    # 总结
    print("  " + "─" * 50)
    print(f"  【RPLD 总 GFLOPs (单帧, batch=1)】: {total_flops/1e9:.2f} GFLOPs")
    print()
    print("=" * 60)
    print()

    # ---- 3. 各阶段 FLOPs 占比 ----
    print("【3. 单帧推理各阶段 FLOPs 拆解】\n")
    print(f"  {'阶段':<28} {'GFLOPs':>10}  {'占比':>8}")
    print(f"  {'─' * 46}")

    stages_data = [
        ("model_s (图片→特征+分割)", wrapper_s, (dummy_img,)),
        ("model_c (分割→系数初始化)", wrapper_c, (dummy_prob,)),
        ("光流对齐 (Flow)", wrapper_flow, (dummy_query_feat, dummy_key_feat)),
        ("水平 BiMamba 扫描", model_main.fusion_mamba, (dummy_seq,)),
        ("垂直 BiMamba 扫描", model_main.fusion_mamba_v, (dummy_seq,)),
        ("final_conv 降维", model_main.final_conv, (dummy_combined,)),
        ("分类头 (Classifier)", wrapper_cls, (dummy_img_feat,)),
        ("回归头+KAN (Regression)", wrapper_reg, (dummy_prob_2ch,)),
        ("空间门控 (SpatialGate)", model_main.spatial_attn, (dummy_gate_input,)),
    ]

    try:
        from thop import profile
        stage_flops = []
        for name, model, inputs in stages_data:
            f, _ = profile(model, inputs=inputs, verbose=False)
            stage_flops.append((name, f / 1e9))
            print(f"  {name:<28} {f/1e9:>10.4f}")

        sum_f = sum(f for _, f in stage_flops)
        print(f"  {'─' * 46}")
        for name, f in stage_flops:
            pct = f / sum_f * 100
            bar = '#' * int(pct / 2) + '.' * (50 - int(pct / 2))
            print(f"  {name:<28} {bar} {pct:>5.1f}%")
    except ImportError:
        pass

    print()
    print("=" * 60)
    print("  分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
