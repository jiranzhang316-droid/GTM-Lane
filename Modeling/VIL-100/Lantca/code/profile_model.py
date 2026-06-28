"""
Lantca 模型参数量 & FLOPs 分析脚本
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
    """主模型：回归头（DeformConv），输入 prob_map (B,2,H,W)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prob_map_2ch):
        self.model.prob_map = prob_map_2ch
        out = self.model.forward_for_regression()
        return out['coeff_map']


class ModelMain_FeatGuideWrapper(nn.Module):
    """主模型：引导特征编码"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, guide_map):
        return self.model.forward_for_guidance_feat(guide_map)


# LSTN + LSAB 包装器
class ModelMain_LSTN_LSAB_Wrapper(nn.Module):
    """
    包装 LSTN + LSAB 的长短期记忆聚合流程（首帧模式）
    流程: img_feat → conv_layer(downsample) → pos_emb → InstanceQueryBank → LSAB → conv_layer2(upsample)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img_feat, guide_cls):
        # 初始化 memory（模拟首帧，无长期记忆）
        self.model.memory = {
            'img_feat': {'t-0': img_feat},
            'guide_cls': {'t-1': guide_cls},
            'long': {'t-1': [], 't-0': []},
            'pos_emb': {},
            'short': {'t-0': []},
        }
        self.model.pos_emb = None
        self.model.lsab_curr_memories = None
        self.model.short_term_memories = []
        self.model.long_term_memories = []
        out = self.model.forward_for_feat_aggregation(is_training=False)
        return self.model.img_feat


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
    print("  Lantca 模型参数量 & FLOPs 分析")
    print("=" * 60)
    print()

    # ---- 配置 ----
    cfg = Config()
    cfg.backbone = '18'
    cfg.num_t = 1
    cfg.window_size = 7
    cfg.use_instance_query_id = True
    cfg.max_lane_num = 6

    H_img = ((cfg.height - cfg.crop_size) // 32) * 32  # 256
    W_img = (cfg.width // 32) * 32                      # 640
    H_feat = cfg.height // cfg.scale_factor['seg'][0]   # 96
    W_feat = cfg.width // cfg.scale_factor['seg'][0]    # 160
    # LSAB 内部尺寸 (conv_layer stride=2 后)
    H_enc = H_feat // 2  # 48
    W_enc = W_feat // 2  # 80

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

    # ---- 主模型各组件 ----
    p_cls, _ = count_params(model_main.classifier, "  - classifier (2分类头)")
    total_params += p_cls

    p_fe, _ = count_params(model_main.feat_embed, "  - feat_embed (特征嵌入)")
    total_params += p_fe

    p_fg, _ = count_params(model_main.feat_guide, "  - feat_guide (引导特征编码)")
    total_params += p_fg

    p_fe2, _ = count_params(model_main.feat_embedding, "  - feat_embedding (回归特征嵌入)")
    total_params += p_fe2

    p_reg, _ = count_params(model_main.regressor, "  - regressor (回归器)")
    total_params += p_reg

    p_off, _ = count_params(model_main.offset_regression, "  - offset_regression (偏移预测)")
    total_params += p_off

    p_dc, _ = count_params(model_main.deform_conv2d, "  - deform_conv2d (可变形卷积)")
    total_params += p_dc

    p_cl, _ = count_params(model_main.conv_layer, "  - conv_layer (下采样 64→128)")
    total_params += p_cl

    p_cl2, _ = count_params(model_main.conv_layer2, "  - conv_layer2 (上采样 128→64)")
    total_params += p_cl2

    p_mp, _ = count_params(model_main.max_pool, "  - max_pool")
    total_params += p_mp

    # LSTN 组件
    p_lstn, _ = count_params(model_main.lstn, "  - lstn (位置编码+ID分配)")
    total_params += p_lstn

    # LSAB 组件（LongShortTermTransformerBlock）
    p_lsab, _ = count_params(model_main.LSAB, "  - LSAB (长短期Transformer)")
    total_params += p_lsab

    # InstanceQueryBank
    if hasattr(model_main, 'instance_query_bank') and model_main.instance_query_bank is not None:
        p_iqb, _ = count_params(model_main.instance_query_bank, "  - InstanceQueryBank (实例查询)")
        total_params += p_iqb

    p_pe, _ = count_params(model_main.pos_generator, "  - pos_generator (正弦位置编码)")
    total_params += p_pe

    print(f"  {'─' * 35}")
    print(f"  【Lantca 总参数量】: {total_params:>12,}  ({total_params/1e6:.2f} M)")
    print()

    # ---- 2. FLOPs 统计 ----
    print("-" * 40)
    print("【2. FLOPs 统计 (输入 batch_size=1)】\n")

    total_flops = 0

    # 2a. model_s: 图片 → 分割图
    dummy_img = torch.randn(1, 3, H_img, W_img).cuda()
    wrapper_s = ModelS_Wrapper(model_s)
    flops, _ = count_flops(wrapper_s, (dummy_img,),
                           f"model_s (图片→分割图) 输入: (1,3,{H_img},{W_img})")
    total_flops += flops

    # 2b. model_c: 分割图 → 系数图
    dummy_prob = torch.randn(1, 1, H_feat, W_feat).cuda()
    wrapper_c = ModelC_Wrapper(model_c)
    flops, _ = count_flops(wrapper_c, (dummy_prob,),
                           f"model_c (分割图→系数图) 输入: (1,1,{H_feat},{W_feat})")
    total_flops += flops

    # 2c. 引导特征编码
    dummy_guide = torch.randn(1, 1, H_feat, W_feat).cuda()
    wrapper_guide = ModelMain_FeatGuideWrapper(model_main)
    flops, _ = count_flops(wrapper_guide, (dummy_guide,),
                           f"feat_guide (引导特征编码) 输入: (1,1,{H_feat},{W_feat})")
    total_flops += flops

    # 2d. LSAB 输入准备: conv_layer 降采样
    dummy_img_feat = torch.randn(1, 64, H_feat, W_feat).cuda()
    flops, _ = count_flops(model_main.conv_layer, (dummy_img_feat,),
                           f"conv_layer (下采样) 输入: (1,64,{H_feat},{W_feat})")
    total_flops += flops

    # 2e. LSTN pos_emb 生成
    dummy_emb_for_pos = torch.randn(1, 128, H_enc, W_enc).cuda()
    flops, _ = count_flops(model_main.lstn.pos_generator, (dummy_emb_for_pos,),
                           f"pos_generator (正弦位置编码) 输入: (1,128,{H_enc},{W_enc})")
    total_flops += flops

    # 2f. InstanceQueryBank
    if hasattr(model_main, 'instance_query_bank') and model_main.instance_query_bank is not None:
        flops, _ = count_flops(model_main.instance_query_bank, (1,),
                               f"InstanceQueryBank (实例查询batch=1) ")
        total_flops += flops

    # 2g. LSTN patch_wise_id_bank (one_hot_mask → id_emb)
    dummy_onehot = torch.randn(1, 2, H_enc, W_enc).cuda()
    flops, _ = count_flops(model_main.lstn.patch_wise_id_bank, (dummy_onehot,),
                           f"lstn.patch_wise_id_bank (2→128) 输入: (1,2,{H_enc},{W_enc})")
    total_flops += flops

    # 2h. LSAB (LongShortTermTransformerBlock) — 核心模块
    # 输入: tgt shape (HW, B, 128) = (48*80, 1, 128) = (3840, 1, 128)
    dummy_lsab_input = torch.randn(H_enc * W_enc, 1, 128).cuda()
    dummy_pos_emb = torch.randn(H_enc * W_enc, 1, 128).cuda()  # (HW, B, C)
    dummy_id_emb = torch.randn(H_enc * W_enc, 1, 128).cuda()
    # LSAB 首帧调用: long_term_memory=None, short_term_memory=None
    flops, _ = count_flops(
        model_main.LSAB,
        (dummy_lsab_input, None, None, dummy_id_emb, dummy_pos_emb, (H_enc, W_enc)),
        f"LSAB (长短期Transformer) 输入: ({H_enc*W_enc},1,128)"
    )
    total_flops += flops

    # 2i. conv_layer2 上采样还原
    dummy_upsampled = torch.randn(1, 128, H_enc, W_enc).cuda()
    flops, _ = count_flops(model_main.conv_layer2, (dummy_upsampled,),
                           f"conv_layer2 (上采样还原) 输入: (1,128,{H_enc},{W_enc})")
    total_flops += flops

    # 2j. 分类头
    wrapper_cls = ModelMain_ClassifierWrapper(model_main)
    flops, _ = count_flops(wrapper_cls, (dummy_img_feat,),
                           f"主模型-分类头 输入: (1,64,{H_feat},{W_feat})")
    total_flops += flops

    # 2k. 回归头（DeformConv，无KAN）
    dummy_prob_2ch = torch.randn(1, 2, H_feat, W_feat).cuda()
    wrapper_reg = ModelMain_RegressionWrapper(model_main)
    flops, _ = count_flops(wrapper_reg, (dummy_prob_2ch,),
                           f"主模型-回归头(DeformConv) 输入: (1,2,{H_feat},{W_feat})")
    total_flops += flops

    # 总结
    print("  " + "─" * 50)
    print(f"  【Lantca 总 GFLOPs (单帧, batch=1)】: {total_flops/1e9:.2f} GFLOPs")
    print()
    print("=" * 60)
    print()

    # ---- 3. 各阶段 FLOPs 占比 ----
    print("【3. 单帧推理各阶段 FLOPs 拆解】\n")
    print(f"  {'阶段':<35} {'GFLOPs':>10}  {'占比':>8}")
    print(f"  {'─' * 53}")

    try:
        from thop import profile

        stages_data = [
            ("model_s (图片→特征+分割)", wrapper_s, (dummy_img,)),
            ("model_c (分割→系数初始化)", wrapper_c, (dummy_prob,)),
            ("feat_guide (引导特征编码)", wrapper_guide, (dummy_guide,)),
            ("conv_layer (下采样 64→128)", model_main.conv_layer, (dummy_img_feat,)),
            ("LSTN pos_emb (正弦位置编码)", model_main.lstn.pos_generator, (dummy_emb_for_pos,)),
            ("lstn patch_wise_id_bank (2→128)", model_main.lstn.patch_wise_id_bank, (dummy_onehot,)),
            ("LSAB (长短期Transformer核心)", model_main.LSAB,
             (dummy_lsab_input, None, None, dummy_id_emb, dummy_pos_emb, (H_enc, W_enc))),
            ("conv_layer2 (上采样还原)", model_main.conv_layer2, (dummy_upsampled,)),
            ("分类头 (Classifier)", wrapper_cls, (dummy_img_feat,)),
            ("回归头 DeformConv (Regression)", wrapper_reg, (dummy_prob_2ch,)),
        ]
        if hasattr(model_main, 'instance_query_bank') and model_main.instance_query_bank is not None:
            stages_data.insert(5, ("InstanceQueryBank (实例查询)", model_main.instance_query_bank, (1,)))

        stage_flops = []
        for name, model, inputs in stages_data:
            f, _ = profile(model, inputs=inputs, verbose=False)
            stage_flops.append((name, f / 1e9))
            print(f"  {name:<35} {f/1e9:>10.4f}")

        sum_f = sum(f for _, f in stage_flops)
        print(f"  {'─' * 53}")
        for name, f in stage_flops:
            pct = f / sum_f * 100
            bar = '#' * int(pct / 2) + '.' * (50 - int(pct / 2))
            print(f"  {name:<35} {bar} {pct:>5.1f}%")
    except ImportError:
        pass

    print()
    print("=" * 60)
    print("  分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
