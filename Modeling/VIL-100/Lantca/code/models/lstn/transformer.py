import torch
import torch.nn.functional as F
from torch import nn

from .basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d, ScaleOffset, mask_out
from .attention import silu, MultiheadAttention, MultiheadLocalAttentionV2, MultiheadLocalAttentionV3, GatedPropagation, LocalGatedPropagation


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")



class LongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model=128,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()
        # 归一化
        self.norm1 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model) #通过线性层得到Q、V
        self.linear_V = nn.Linear(d_model, d_model)

        self.token_embedding = nn.Parameter(torch.zeros(1, 128, 48, 80))  #初始化一个跟输入一摸一样的零矩阵
        self.token_t = None

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model, att_nhead, dilation=local_dilation, use_linear=False, dropout=st_dropout)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor+pos

    def forward(self, tgt, long_term_memory=None, short_term_memory=None, curr_id_emb=None,
                self_pos=None, size_2d=(30, 30)):

        # Self-attention 
        _tgt = self.norm1(tgt)  
        q = k = self.with_pos_embed(_tgt, self_pos) 
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0] 
        tgt = tgt + self.droppath(tgt2) 

        _tgt = self.norm2(tgt) 

        curr_Q = self.linear_Q(_tgt) 
        curr_K = curr_Q 
        curr_V = _tgt 

        local_Q = seq_to_2d(curr_Q, size_2d) 
        b = local_Q.size(0)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(curr_K, curr_V, curr_id_emb)  #将v里面加入位置向量
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
            token = self.token_embedding.expand(b, -1, -1, -1)#可学习的嵌入，它在批次中进行扩展，用于提供额外的上下文信息。
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
            token = seq_to_2d(self.token_t, size_2d)

        tgt2 = self.short_term_attn(token, local_K, local_V)[0]   #以初始化的token作为初始查询得到加权后的特征表示
        self.token_t = tgt2
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]  #以实际的Q查询得到加权后的特征表示
        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3) #融合到初始的特征中，得到三合一特征

        # Feed-forward
        _tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))
        tgt = tgt + self.droppath(tgt2)       
        # tgt是经过自注意力、短期注意力、长短期记忆融合 和 前馈网络
        #[curr_K, curr_V] 通过当前的查询（q）计算得到的键和值是用于自注意力的
        #[global_K, global_V]长期记忆的键和值
        #[local_K, local_V]]短期记忆的键和值
        return tgt, [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        K = key
        V = self.linear_V(value + id_emb)
        return K, V    

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


