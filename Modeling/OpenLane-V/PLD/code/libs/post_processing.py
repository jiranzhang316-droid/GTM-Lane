import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *

class Post_Processing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg'] #值为4 缩放因子

        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def draw_polyline_cv(self, pts, color=(255, 0, 0), s=1):
        out = np.ascontiguousarray(np.zeros(self.seg_map.shape, dtype=np.float32))
        out = cv2.polylines(out, pts, False, color, s)
        return out

    def measure_confidence_score(self, seg_map, lane_mask):
        lane_mask[:self.height_idx[0]] = 0 #去掉顶部无效区域
        score = np.sum(lane_mask * seg_map) / (np.sum(lane_mask) + 1e-8)  #求平均置信度
        return score
    # RVLD 里从 dense seg + coeff 中抽取“实例车道”的核心 NMS 逻辑
    def run_for_nms(self):
        seg_map = to_np(self.seg_map) ## [H, W] 车道概率图
        coeff_map = self.coeff_map.clone()
        h, w = seg_map.shape

        if self.mode == 'init': #第一帧使用0.6的阈值
            nms_thresd = 0.6
        elif self.mode == 'f':
            nms_thresd = self.cfg.nms_thresd
        for i in range(self.cfg.max_lane_num * 2): #最多尝试 2 × max_lane_num 次，防止死循环
            idx_max = np.argmax(seg_map) #找当前最强响应点
            if len(self.out['idx']) >= self.cfg.max_lane_num: #超过最大车道数就停
                break
            if seg_map[idx_max // w, idx_max % w] > nms_thresd: # 首先得到一个坐标，然后再去查概率 判断这个最强响应点的阈值是不是大于0.6
                coeff = coeff_map[:, idx_max // w, idx_max % w] #如果是大于0.6，再取出对应的车道线系数
                # removal
                x_coords = torch.matmul(self.U, coeff) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2 #u是预定义的基矩阵，乘预测的coeff得到车道线 100个点
                x_coords = x_coords / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1) #做尺度映射
                y_coords = to_tensor(self.cfg.py_coord).view(1, len(x_coords), 1) / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1) #再生成y
                x_coords = x_coords.view(1, len(x_coords), 1)
                lane_coords = torch.cat((x_coords, y_coords), dim=2)#得到预测的标签坐标
                lane_mask = self.draw_polyline_cv(np.int32(to_np(lane_coords)), color=(1, 1, 1), s=self.cfg.removal['lane_width']) #画 mask 粗线，用于抑制
                lane_mask2 = self.draw_polyline_cv(np.int32(to_np(lane_coords)), color=(1, 1, 1), s=1) #画 mask 细线，用于 score 计算
                score = self.measure_confidence_score(seg_map, lane_mask2) #计算置信度  看这条线经过的地方，在 seg_map 上整体有多可信。
                seg_map[idx_max // w, idx_max % w] = 0  #把当前这个已经处理的做大值点置为0
                if score >= 0.3: #如果这条 lane 可信就保存
                    self.out['idx'].append(int(idx_max)) #像素点的位置
                    self.out['coeff'].append(coeff.view(-1, 1)) #对应的车道线系数
                    seg_map = seg_map * (1 - lane_mask) #NMS 抑制邻域 也就是已经画出车道线的部分 再原图中将概率置为0
                    self.out['lane_pts'].append(np.int32(to_np(lane_coords))) # 收集车道点坐标 用来画车道线的坐标
            else:

                break
    # 这一段是 把已经选出来的 lane coeff → 转成最终用于输出/监督的 x 坐标序列
    def run_for_coeff_to_x_coord_conversion(self):
        x_coords = list()
        # self.out['coeff'] 是刚才 run_for_nms 里收集的可能是车道线的车道线系数
        coeff_results = self.out['coeff']
        if len(coeff_results) != 0: #拼接所有 lane
            coeff_results = torch.cat(coeff_results, dim=1) #拼接所有 [top_m, num_lanes]
        if len(coeff_results) != 0:
            x_coords = torch.matmul(self.U, coeff_results) #coeff → x 得到 [num_y, num_lanes]
            x_coords = x_coords * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2 #因为网络回归的是 [-1,1] 空间，需要拉回真实像素 反归一化
            x_coords = x_coords.permute(1, 0) #维度换位 [num_lanes, num_y] 例如车道线1【x1，x2，x3】

        self.x_coords = x_coords
        return {'x_coords': x_coords}

    # # 根据 seg_map，找出车道在 y 方向上的 起始高度 idx_st 和结束高度 idx_ed，用于后面只在有效区域生成 lane points
    def run_for_height_filtering(self):
        idxlist = to_np(torch.sum((self.seg_map > self.cfg.height_thresd), dim=1)).nonzero()[0] #统计每一行有没有车道 以0.5为阈值划分0/1 # 每一行统计有多少像素属于车道
        if len(idxlist) > 0: #这个图片存在车道，就计算车道起止点 idxlist[0]是起点  idxlist[-1]是终点
            idx_ed = idxlist[0] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1) #因为预测结果都是下采样过的所以要映射会原来的车长度
            idx_st = idxlist[-1] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            lane_idx_ed = np.argmin(np.abs(self.cfg.py_coord - idx_ed)) #对齐根据真实标签采样的标准的网络固定采样的 y 坐标位置，找到距离最近的起止点
            lane_idx_st = np.argmin(np.abs(self.cfg.py_coord - idx_st))
            self.height_idx = [lane_idx_ed, lane_idx_st] #作为起止车道线点  注意这里并不是图片里面的车道线的区间，而是这个区域的图像有车道线
            return {'height_idx': [lane_idx_ed, lane_idx_st]} #lane_idx_ed lane_idx_st里面存的是最像真实标签的起止点
        else:
            self.height_idx = None
            return {'height_idx': None}

    def measure_IoU(self, X1, X2):
        ep = 1e-7
        X = X1 + X2
        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)
        iou = X_inter / (X_uni + ep)
        return iou
    # 用预测 lane 结果 + GT 构造一个 guide mask 来“指导”后续分类学习
    def lane_mask_generation_for_training(self, data, gt):
        out = dict()
        out['guide_cls'] = list()  #存每条 lane 的 mask
        out['guide_num'] = list()  #
        h, w = self.seg_map.shape  #当前特征图尺寸
        for i in range(len(data)): #data[i] 表示第 i 条车道
            if len(data[i]['lane_pts']) > 0: #只要不为空就画车道线
                lane_mask = self.draw_polyline_cv(data[i]['lane_pts'], color=(1, 1, 1), s=1) #draw_polyline_cv的作用是把车道坐标点转化到0-1的车道像素掩码
                lane_mask = cv2.dilate(lane_mask, kernel=(3, 3), iterations=1)

            else:
                lane_mask = np.zeros(shape=self.seg_map.shape, dtype=np.float32)
            height_idx = data[i]['height_idx']
            if height_idx is not None: #根据 height_idx 截断上半部分 只监督有效区域
                h_idx1 = int(self.cfg.py_coord[np.minimum(height_idx[0] + 1, len(self.cfg.py_coord) - 1)] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1))
                lane_mask[:h_idx1] = 0
            lane_mask = to_tensor(lane_mask).view(1, h, w)
            gt_lane_mask = gt['seg_label'][self.sf[0]][i].view(1, h, w).type(torch.float32)
            iou = self.measure_IoU(lane_mask, gt_lane_mask)  #判断预测车道和真实车道的合并多少
            if iou > 0.6: #如果是大于0.6 就认为预测车道就是真实车道
                lane_mask = lane_mask
            else:
                case = random.randint(0, 1) #人为加干扰区域
                if case == 0:
                    lane_mask = gt_lane_mask
                else:
                    lane_mask = torch.zeros(size=(1, h, w), dtype=torch.float32).cuda()
            case_neg = random.randint(0, 1)
            if case_neg == 1:
                lane_mask += gt['guide_mask_neg'][i].view(1, h, w)
            lane_mask_f = ((lane_mask) != 0).type(torch.float32)  #将mask进行二值化
            out['guide_cls'].append(lane_mask_f)
        # lane_mask 由 100 个预测点连接成的连续车道区域 mask
        lane_mask = torch.cat(out['guide_cls'])  #拼接，最终得到[B, 1, H, W]
        b, h, w = lane_mask.shape
        out['guide_cls'] = lane_mask.view(b, 1, h, w)
        return out

    def lane_mask_generation(self, data):
        out = dict()
        out['guide_cls'] = list()
        out['guide_num'] = list()

        for i in range(len(data)):
            if len(data[i]['lane_pts']) > 0:
                lane_mask = self.draw_polyline_cv(data[i]['lane_pts'], color=(1, 1, 1), s=1)
                lane_mask = cv2.dilate(lane_mask, kernel=(3, 3), iterations=1)

            else:
                lane_mask = np.zeros(shape=self.seg_map.shape, dtype=np.float32)
            height_idx = data[i]['height_idx']
            if height_idx is not None:
                h_idx1 = int(self.cfg.py_coord[np.minimum(height_idx[0] + 1, len(self.cfg.py_coord) - 1)] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1))
                lane_mask[:h_idx1] = 0
            out['guide_cls'].append(lane_mask)
            out['guide_num'].append(len(data[i]['lane_pts']))

        lane_mask = to_tensor(np.array(out['guide_cls']))
        b, h, w = lane_mask.shape
        out['guide_cls'] = lane_mask.view(b, 1, h, w)
        return out

    def run_for_test(self, data, batch_idx):
        # results
        out_f = list()

        if self.mode == 'init':
            self.seg_map = data['seg_map_init'][batch_idx, 0]
            self.coeff_map = data['coeff_map_init'][batch_idx]
        elif self.mode == 'f':
            self.seg_map = data['seg_map'][0, 0]
            self.coeff_map = data['coeff_map'][0]

        self.out = dict()
        self.out['coeff'] = list()
        self.out['lane_pts'] = list()
        self.out['idx'] = list()
        self.out['height'] = list()
        # 根据 seg_map，找出车道在 y 方向上的 起始高度 idx_st 和结束高度 idx_ed，用于后面只在有效区域生成 lane points
        self.out.update(self.run_for_height_filtering())
        self.run_for_nms()
        self.out.update(self.run_for_coeff_to_x_coord_conversion())

        out_f.append(self.out)

        return out_f

    def run_for_training(self, data):
        # results
        out_f = list()
        b = len(data['seg_map_init']) if self.mode == 'init' else len(data['seg_map']) #根据模型是不是init判断batch_size

        for i in range(b):
            if self.mode == 'init':
                self.seg_map = data['seg_map_init'][i, 0] #这个就是seg模型得到的车道线的概率 取出宽 高
                self.coeff_map = data['coeff_map_init'][i] #取出 [top_m, H, W]
            elif self.mode == 'f':
                self.seg_map = data['seg_map'][i, 0]
                self.coeff_map = data['coeff_map'][i]

            self.out = dict()
            self.out['coeff'] = list()
            self.out['lane_pts'] = list()
            self.out['idx'] = list()
            self.out['height'] = list()
            self.out.update(self.run_for_height_filtering()) #一句话概括，是用来切割有车道线的图片的
            self.run_for_nms() # 找到车道线的点 最多找6个
            self.out.update(self.run_for_coeff_to_x_coord_conversion())

            out_f.append(self.out)

        return out_f

    def update(self, mode=None):
        self.mode = mode