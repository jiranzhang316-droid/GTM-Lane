import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *

from .lstn.transformer import LongShortTermTransformerBlock     
from .lstn.lstn import LSTN
from .lstn.position import PositionEmbeddingSine
from .lstn.basic import one_hot_mask, seq_to_2d

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

class InstanceQueryBank(nn.Module):
    """
    Query-based instance identity embedding.
    Replaces the original 2-class foreground/background identity with
    max_lane_num learnable instance queries, each responsible for a lane instance.
    """
    def __init__(self, num_queries, d_model, height, width):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.height = height
        self.width = width
        # Each lane instance has a learnable query vector
        self.instance_queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        # Project each query to a spatial weight map over the feature grid
        self.spatial_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, height * width)
        )
        # Optional: a light fusion layer to blend query embedding with spatial bias
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, batch_size):
        # instance_queries: [N, C]
        # spatial_weights: [N, HW]
        spatial_weights = torch.softmax(self.spatial_proj(self.instance_queries), dim=-1)
        # Weighted sum of queries over spatial locations: [HW, C]
        id_emb = spatial_weights.t() @ self.instance_queries
        id_emb = self.fusion(id_emb)
        # Expand to [HW, B, C]
        id_emb = id_emb.unsqueeze(1).expand(-1, batch_size, -1)
        return id_emb


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

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )

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

        self.lstn = LSTN()
        self.LSAB = LongShortTermTransformerBlock()             
        self.pos_emb = None
        self.pos_generator = PositionEmbeddingSine(128, normalize=True)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # ---------- Instance Query ID Bank ----------
        if self.cfg.use_instance_query_id:
            # Derive LSAB feature spatial size from config
            # img_feat is typically 1/4 resolution, conv_layer stride=2 => 1/8
            enc_h = self.cfg.height // (self.seg_sf[0] * 2)
            enc_w = self.cfg.width // (self.seg_sf[0] * 2)
            self.instance_query_bank = InstanceQueryBank(
                num_queries=self.cfg.max_lane_num,
                d_model=128,
                height=enc_h,
                width=enc_w
            )
        else:
            self.instance_query_bank = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None
        self.pos_emb = None


    def forward_for_mask_generation(self, query_data, key_data):
        data_combined = torch.cat((query_data, key_data), dim=1)
        mask = self.mask_generator(data_combined)
        mask = torch.sigmoid(mask)
        return mask

    def update_long_term_memory_model(self, new_memories, old_memories):
        """
        Controlled long-term memory update with three modes:
        - 'cat': original concatenation (unbounded growth)
        - 'momentum': exponential moving average (constant memory)
        - 'fixed_pool': concatenation with max length truncation
        """
        if old_memories is None:
            return new_memories
        mode = getattr(self.cfg, 'lt_memory_mode', 'cat')
        if mode == 'cat':
            updated = []
            for new_m, old_m in zip(new_memories, old_memories):
                pair = []
                for n, o in zip(new_m, old_m):
                    if n is None or o is None:
                        pair.append(n if o is None else o)
                    else:
                        pair.append(torch.cat([n, o], dim=0))
                updated.append(pair)
            return updated
        elif mode == 'momentum':
            momentum = getattr(self.cfg, 'lt_momentum', 0.9)
            updated = []
            for new_m, old_m in zip(new_memories, old_memories):
                pair = []
                for n, o in zip(new_m, old_m):
                    if n is None or o is None:
                        pair.append(n if o is None else o)
                    else:
                        pair.append(momentum * o + (1 - momentum) * n)
                updated.append(pair)
            return updated
        elif mode == 'fixed_pool':
            max_len = getattr(self.cfg, 'lt_memory_max_len', 50)
            updated = []
            for new_m, old_m in zip(new_memories, old_memories):
                pair = []
                for n, o in zip(new_m, old_m):
                    if n is None or o is None:
                        pair.append(n if o is None else o)
                        continue
                    combined = torch.cat([n, o], dim=0)
                    limit = max_len * n.size(0)
                    if combined.size(0) > limit:
                        combined = combined[:limit]
                    pair.append(combined)
                updated.append(pair)
            return updated
        else:
            return old_memories

    def forward_for_feat_aggregation(self, is_training=False):       

        key_t = f't-0'         
        query_img_feat = self.memory['img_feat'][key_t]         

        curr_emb = self.conv_layer(query_img_feat)      #变换了一下通道数
        batch_size = query_img_feat.size()[0]
        enc_hw = curr_emb.size()[2]*curr_emb.size()[3] #得到图像大小
        self.size_2d = curr_emb.size()[2:]

        lane_mask=self.memory['guide_cls']['t-1']       #取出t-1时刻的车道掩码
        curr_one_hot_mask = one_hot_mask(lane_mask, 1)
        if not self.memory['long']['t-1']:        
            self.pos_emb = self.lstn.get_pos_emb(curr_emb)  \
                    .expand(batch_size, -1, -1,-1).view(batch_size, -1, enc_hw).permute(2, 0, 1) 
            # ---------- Instance Query ID (improvement 2) ----------
            if self.cfg.use_instance_query_id and self.instance_query_bank is not None:
                curr_id_emb = self.instance_query_bank(batch_size)
            else:
                curr_id_emb = self.lstn.assign_identity(curr_one_hot_mask, batch_size, enc_hw)
            self.memory['pos_emb']['t-0'] = self.pos_emb
        else:          
            curr_id_emb=None   
            self.memory['pos_emb']['t-0']=self.memory['pos_emb']['t-1']  
            self.update_short_term_memory(self.lsab_curr_memorie, batch_size, enc_hw)   
        
        n, c, h, w = curr_emb.size()
        _curr_emb = curr_emb.view(n, c, h*w).permute(2, 0, 1)

        self.long_term_memories=self.memory['long']['t-1']

        lsab_embs, lsab_memories = self.LSAB(_curr_emb, self.long_term_memories, self.short_term_memories, curr_id_emb, self.memory['pos_emb']['t-0'], self.size_2d)      
        self.lsab_curr_memories, lsab_long_memories, lsab_short_memories = lsab_memories      
        self.lsab_curr_memorie = lsab_embs

        img_feat_cur = lsab_embs.permute(1, 2, 0).view(n, c, h, w)       
        self.img_feat = self.conv_layer2(img_feat_cur) #在这里得到新的img_feat

        # ---------- Long-term memory control (improvement 1) ----------
        if not self.memory['long']['t-0']:
            self.memory['long']['t-0'] = lsab_long_memories
        else:
            self.memory['long']['t-0'] = self.update_long_term_memory_model(
                lsab_long_memories, self.memory['long']['t-1']
            )
        self.short_term_memories = self.lsab_curr_memories
        self.memory['short']['t-0'] = self.lsab_curr_memories

        return {'key_probmap': lane_mask,         
                'key_guide': lane_mask,
                'aligned_key_probmap': lane_mask,
                'aligned_key_guide': lane_mask,
                'grid': None,
                'long':self.long_term_memories,
                'pos_emb':self.pos_emb,
                'short':self.short_term_memories
                }


    def forward_for_guidance_feat(self, guide_map):
        feat_guide = self.feat_guide(guide_map)
        return feat_guide

    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        self.prob_map = F.softmax(out, dim=1)
        return {'seg_map_logit': out,
                'seg_map': self.prob_map[:, 1:2]}

    def forward_for_regression(self):
        b, _, _, _ = self.prob_map.shape
        feat_c = self.feat_embedding(self.prob_map[:, 1:].detach())       
        feat_c = feat_c + self.pe.expand(b, -1, -1, -1)     
        offset = self.offset_regression(feat_c)       
        x = self.regressor(feat_c)                 
        coeff_map = self.deform_conv2d(x, offset) 

        return {'coeff_map': coeff_map}

    def update_short_term_memory(self, lsab_curr_memorie, batch_size, enc_hw):

        _lsab_curr_memorie = self.LSAB.norm2(lsab_curr_memorie) 

        curr_Q = self.LSAB.linear_Q(_lsab_curr_memorie) 
        curr_K = curr_Q  
        curr_V = _lsab_curr_memorie 

        [curr_K, curr_V] = [
            seq_to_2d(curr_K, self.size_2d),
            seq_to_2d(curr_V, self.size_2d)
        ]

        self.short_term_memories = [curr_K, curr_V]

