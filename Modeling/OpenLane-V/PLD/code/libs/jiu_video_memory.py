import numpy as np
from libs.utils import *

class Video_Memory(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward_for_dict_initialization(self):
        # 用于为后续视频做记忆处理做准备 图片特征 seg分类的概率map  coeff回归的结果 guide_cls lane mask指导
        self.keylist = ['img_feat', 'prob_map', 'coeff_map', 'guide_cls']
        self.data = dict() #创建字段用来存储
        for key in self.keylist:
            self.data[key] = dict()
        self.memory_t = 0  #计数器 表示当前memory中有多少帧已经存储

    def forward_for_dict_memorization(self):
        # 把已有帧的数据向后移动一格，为下一帧腾出“t-0”的位置
        for i in range(self.memory_t - 1, -1, -1):
            for key in self.keylist:
                self.data[key][f't-{i+1}'] = self.data[key][f't-{i}']  #完后移动一个

        for key in self.keylist:
            self.data[key].pop('t-0') #删除旧数据
        if self.memory_t >= self.cfg.num_t:
            self.memory_t -= 1

    def forward_for_dict_initialization_per_frame(self, t):
        for key in self.keylist:
            self.data[key][t] = dict()
        self.t = t

    def forward_for_dict_update_per_frame(self, model, batch_idx=None, mode=None):
        if mode == 'intra' and batch_idx is not None:
            self.data['img_feat'][self.t] = model.img_feat[batch_idx:batch_idx + 1]
            self.data['prob_map'][self.t] = model.prob_map[batch_idx:batch_idx + 1]
            self.memory_t += 1
        elif mode == 'intra' and batch_idx is None:
            self.data['img_feat'][self.t] = model.img_feat   #将seg分类层之前的向量拿出来 并标注时间序号 t
            self.data['prob_map'][self.t] = model.prob_map   #将经过分类之后且经过sigmid激活后的概率也拿出来并标记
            self.memory_t += 1 #时间步加1
        elif mode == 'update':
            self.data['img_feat'][self.t] = model.img_feat.detach()
            self.data['prob_map'][self.t] = model.prob_map.detach()

    def forward_for_dict_transfer(self, model):
        model.memory = dict()
        for key in self.keylist:
            model.memory[key] = self.data[key]

        return model
