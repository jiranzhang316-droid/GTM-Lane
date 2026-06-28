import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *


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


class LaneQueryAttention(nn.Module):
    """
    Pure-PyTorch multi-head attention (replaces nn.MultiheadAttention).
    Avoids potential device-placement bugs in certain PyTorch versions.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, need_weights=False):
        # Q: [B, Lq, D], K: [B, Lk, D], V: [B, Lv, D]
        B, nq, _ = Q.shape
        _, nk, _ = K.shape

        # Linear projections and reshape to [B, num_heads, len, d_k]
        q = self.W_q(Q).view(B, nq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, nk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, nk, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        # Concatenate heads: [B, num_heads, nq, d_k] -> [B, nq, d_model]
        out = out.transpose(1, 2).contiguous().view(B, nq, self.d_model)
        out = self.W_o(out)

        if need_weights:
            # Average attention weights over heads: [B, num_heads, nq, nk] -> [B, nq, nk]
            attn_weights = attn_weights.mean(dim=1)
            return out, attn_weights
        return out, None


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

        # ========================== 1. Query-based Temporal Aggregation ==========================
        # Instead of warping entire dense feature maps, we propagate N sparse Instance Queries.
        # Each query learns to track one lane instance across frames (DETR-style + Markov chain).
        # =========================================================================================
        self.num_queries = cfg.max_lane_num          # N: max number of lanes (e.g. 6 for VIL-100)
        self.query_dim = 256                         # Dimension of each query vector

        # --- 1.1 Learnable query embeddings (like DETR) ---
        # These are the "empty queries" before the first frame activates them.
        self.query_embed = nn.Embedding(self.num_queries, self.query_dim)

        # --- 1.2 Image feature projection for Cross-Attention K, V ---
        # img_feat is 64-dim, we project to query_dim (256) for attention compatibility.
        self.feat_proj_k = nn.Conv2d(self.c_feat, self.query_dim, kernel_size=1)
        self.feat_proj_v = nn.Conv2d(self.c_feat, self.query_dim, kernel_size=1)

        # --- 1.3 Query Self-Attention (lanes talk to each other) ---
        # Reason: lane lines have strong structural relationships (parallel, converging, etc.)
        self.query_self_attn = LaneQueryAttention(d_model=self.query_dim, num_heads=8, dropout=0.1)

        # --- 1.4 Cross-Attention: Query attends to current frame image features ---
        # This is the core: Track Query (Q) searches relevant regions in image (K, V).
        self.query_cross_attn = LaneQueryAttention(d_model=self.query_dim, num_heads=8, dropout=0.1)

        # --- 1.5 FFN for query refinement ---
        self.query_ffn = nn.Sequential(
            nn.Linear(self.query_dim, self.query_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.query_dim * 4, self.query_dim)
        )

        # --- 1.6 Layer Norms (Pre-norm style for stability) ---
        self.query_norm1 = nn.LayerNorm(self.query_dim)
        self.query_norm2 = nn.LayerNorm(self.query_dim)
        self.query_norm3 = nn.LayerNorm(self.query_dim)

        # --- 1.7 Temporal Gate (Markov Chain fusion) ---
        # Controls how much previous frame's query memory is preserved.
        # gate = sigmoid(MLP([curr_query; prev_query]))
        # new_query = gate * prev_query + (1 - gate) * curr_query
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.query_dim * 2, self.query_dim),
            nn.Sigmoid()
        )

        # --- 1.8 Existence prediction head (Birth / Death mechanism) ---
        # Each query predicts whether its tracked lane still exists in current frame.
        self.exist_head = nn.Linear(self.query_dim, 1)

        # --- 1.9 Query -> Dense feature rendering (for backward compatibility) ---
        # Although we use sparse queries for temporal propagation, we still need
        # dense feature maps to feed the original classifier & regressor.
        # We "render" queries back to spatial features using cross-attention weights.
        self.query_to_feat = nn.Sequential(
            nn.Linear(self.query_dim, self.c_feat),
            nn.ReLU(),
            nn.Linear(self.c_feat, self.c_feat)
        )

        # Spatial modulation: learnable scale for query-rendered features
        self.query_render_scale = nn.Parameter(torch.zeros(1))

        # ========================== 2. Original PLD modules (preserved) ==========================
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

        self.pe = positionalencoding2d(
            d_model=self.c_feat,
            height=self.cfg.height // self.seg_sf[0],
            width=self.cfg.width // self.seg_sf[0]
        ).cuda()

        # Buffer for passing queries to Video_Memory between frames
        self.query_output = dict()

    def grid_generator(self):
        x = np.linspace(0, self.cfg.width // self.seg_sf[0] - 1, self.cfg.width // self.seg_sf[0])
        y = np.linspace(0, self.cfg.height // self.seg_sf[0] - 1, self.cfg.height // self.seg_sf[0])
        grid_xy = np.float32(np.meshgrid(x, y))
        _, h, w = grid_xy.shape
        self.grid_xy = to_tensor(grid_xy).permute(1, 2, 0).view(1, h, w, 2)
        self.grid_xy[:, :, :, 0] = (self.grid_xy[:, :, :, 0] / (self.cfg.width // self.seg_sf[0] - 1) - 0.5) * 2
        self.grid_xy[:, :, :, 1] = (self.grid_xy[:, :, :, 1] / (self.cfg.height // self.seg_sf[0] - 1) - 0.5) * 2

    def init_queries(self, batch_size, device):
        """
        Initialize N learnable queries for the first frame (DETR-style).
        Shape: [B, N, D]
        During training, these queries learn to specialize on different lane instances.
        """
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return queries.to(device)

    def update_queries_temporal(self, queries, prev_queries):
        """
        Markov chain update: fuse current frame query update with previous track query.

        Args:
            queries:      [B, N, D] - updated by cross-attention in current frame
            prev_queries: [B, N, D] - track query inherited from previous frame
        Returns:
            updated_queries: [B, N, D]
        """
        gate = self.temporal_gate(torch.cat([queries, prev_queries], dim=-1))
        return gate * prev_queries + (1.0 - gate) * queries

    def forward_for_feat_aggregation(self, is_training=False):
        """
        Query-based temporal aggregation (REPLACES dense optical flow warping + CNN fusion).

        Original PLD workflow (DENSE):
            For each previous frame t:
                1. Compute optical flow between current and previous frame
                2. Warp entire feature map using grid_sample
                3. Concat [current, warped_history, guide] -> CNN fusion

        New workflow (SPARSE - Query Propagation):
            1. Inherit previous frame's Track Queries (N x D vectors)
               OR initialize empty learnable queries for first frame.
            2. Cross-Attention: Track Queries (Q) attend to current frame features (K, V).
               -> Each query updates itself by looking at relevant image regions.
            3. Self-Attention: Queries exchange information (lane structure reasoning).
            4. Temporal Gate: Fuse with previous query memory (Markov chain).
            5. Birth/Death: Low-confidence queries are replaced by fresh queries.
            6. Render queries back to dense feature map for classifier & regressor.
        """
        key_t_0 = 't-0'
        curr_feat = self.memory['img_feat'][key_t_0]  # [B, 64, H, W]
        B = curr_feat.size(0)
        device = curr_feat.device

        # ------------------------------------------------------------------
        # Step 1: Initialize or inherit Track Queries
        # ------------------------------------------------------------------
        prev_queries = None
        if 'queries' in self.memory and 't-1' in self.memory['queries']:
            # Markov chain: inherit previous frame's Track Query
            candidate = self.memory['queries']['t-1']
            # video_memory initializes entries as empty dict() before actual save.
            # Only accept if it is a real tensor (saved by mode='update').
            if isinstance(candidate, torch.Tensor):
                prev_queries = candidate  # [B, N, D]

        if prev_queries is None:
            # First frame: start from learnable query embeddings
            queries = self.init_queries(B, device)
            is_first_frame = True
        else:
            # Defense: ensure prev_queries is on the same device as curr_feat
            # (Video_Memory stores tensors by reference; explicit .to() prevents mismatch)
            queries = prev_queries.to(device).clone()
            is_first_frame = False

        # ------------------------------------------------------------------
        # Step 2: Project image features to K, V for cross-attention
        # ------------------------------------------------------------------
        feat_k = self.feat_proj_k(curr_feat)  # [B, D, H, W]
        feat_v = self.feat_proj_v(curr_feat)  # [B, D, H, W]

        # Flatten spatial dimensions: [B, D, H, W] -> [B, HW, D]
        feat_k = feat_k.flatten(2).permute(0, 2, 1)
        feat_v = feat_v.flatten(2).permute(0, 2, 1)

        # ------------------------------------------------------------------
        # Step 3: Cross-Attention
        #   Query: Track Queries [B, N, D] (what we want to update)
        #   Key:   Image features [B, HW, D] (where to look)
        #   Value: Image features [B, HW, D] (what information to gather)
        #
        # Each query will attend to spatial regions that correspond to its lane.
        # attn_weights: [B, N, HW] tells us which pixels each query is "looking at".
        # ------------------------------------------------------------------
        attn_out, attn_weights = self.query_cross_attn(queries, feat_k, feat_v, need_weights=True)
        queries = self.query_norm1(queries + attn_out)

        # ------------------------------------------------------------------
        # Step 4: Self-Attention (Query-Query interaction)
        # Lane lines are not independent: they are parallel, evenly spaced, etc.
        # Self-attention lets queries share geometric context.
        # ------------------------------------------------------------------
        self_out, _ = self.query_self_attn(queries, queries, queries)
        queries = self.query_norm2(queries + self_out)

        # ------------------------------------------------------------------
        # Step 5: FFN (per-query non-linear transformation)
        # ------------------------------------------------------------------
        ffn_out = self.query_ffn(queries)
        queries = self.query_norm3(queries + ffn_out)

        # ------------------------------------------------------------------
        # Step 6: Temporal Gate (Markov Chain)
        # Instead of storing entire feature maps, we only keep N vectors.
        # The gate decides how much historical memory vs. current observation.
        # ------------------------------------------------------------------
        if not is_first_frame and prev_queries is not None:
            # Ensure prev_queries is on the same device before temporal fusion
            prev_queries = prev_queries.to(device)
            queries = self.update_queries_temporal(queries, prev_queries)

        # ------------------------------------------------------------------
        # Step 7: Birth / Death mechanism
        # Each query predicts existence probability.
        # Dead queries (exist_prob < thresh) are reborn as fresh queries.
        # This handles lanes entering/leaving the field of view.
        # ------------------------------------------------------------------
        exist_logits = self.exist_head(queries).squeeze(-1)  # [B, N]
        exist_prob = torch.sigmoid(exist_logits)

        DEATH_THRESH = 0.3
        if not is_first_frame:
            dead_mask = exist_prob < DEATH_THRESH  # [B, N]
            if dead_mask.any():
                new_queries = self.init_queries(B, device)
                dead_mask = dead_mask.unsqueeze(-1).expand(-1, -1, self.query_dim)
                queries = torch.where(dead_mask, new_queries, queries)

        # Save current queries as track queries for the NEXT frame.
        # Video_Memory will pick this up via model.query_output.
        self.query_output['t-0'] = queries.detach()
        self.query_output['exist_prob'] = exist_prob.detach()
        self.query_output['exist_logits'] = exist_logits.detach()

        # ------------------------------------------------------------------
        # Step 8: Render queries back to dense feature map
        # ------------------------------------------------------------------
        # Although temporal propagation is sparse, the downstream classifier
        # and regressor still expect dense feature maps [B, C, H, W].
        # We "spray" query information back to spatial locations using
        # the cross-attention weights (each query contributes to where it looked).
        # ------------------------------------------------------------------
        # Get actual spatial dimensions from curr_feat (NOT assuming square!)
        # curr_feat: [B, 64, H, W] where H=96, W=160 for VIL-100 (384/4, 640/4)
        _, _, H, W = curr_feat.shape
        HW = H * W

        # attn_weights: [B, N, HW] -> softmax over spatial for each query
        # We use these as spatial assignment weights.
        attn_weights_norm = F.softmax(attn_weights, dim=-1)  # [B, N, HW]

        # Project queries to feature dimension
        q_feat = self.query_to_feat(queries)  # [B, N, 64]

        # Distribute query features to spatial locations:
        #   weighted_feat[b, hw, c] = sum_n attn_weights[b, n, hw] * q_feat[b, n, c]
        weighted_feat = torch.bmm(attn_weights_norm.permute(0, 2, 1), q_feat)  # [B, HW, 64]
        weighted_feat = weighted_feat.permute(0, 2, 1).view(B, self.c_feat, H, W)

        # Modulate original feature with query-rendered enhancement.
        # Use a learnable scale initialized near 0 (conservative at start).
        scale = torch.sigmoid(self.query_render_scale.to(device))
        self.img_feat = curr_feat + scale * weighted_feat

        # ------------------------------------------------------------------
        # Step 9: Return compatible dict
        # We preserve the original key structure so train.py / loss.py don't break.
        # For fields no longer used in sparse paradigm, we return zero tensors
        # with the same shape as original PLD (avoiding NoneType subscript errors).
        # ------------------------------------------------------------------
        key_probmap = self.memory['prob_map'][f't-{self.cfg.num_t}'][:, 1:]
        key_guide = self.memory['guide_cls'][f't-{self.cfg.num_t}']

        # Dummy tensors matching original dense-warping shapes (visualizer expects non-None)
        aligned_key_probmap = torch.zeros_like(key_probmap)
        aligned_key_guide = torch.zeros_like(key_guide)
        grid = self.grid_xy.expand(B, -1, -1, -1).clone().to(device)

        return {
            'key_probmap': key_probmap,
            'key_guide': key_guide,
            'aligned_key_probmap': aligned_key_probmap,
            'aligned_key_guide': aligned_key_guide,
            'grid': grid,
            'exist_logits': exist_logits,
            'exist_prob': exist_prob,
            'attn_weights': attn_weights,
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
