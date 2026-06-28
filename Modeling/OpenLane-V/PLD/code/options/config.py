import os
import torch

import numpy as np
# 全局的控制中心
class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        self.setting_for_visualization()
        self.setting_for_save()
        # --------preprocessing-------- #
        self.setting_for_preprocessing()
        # --------modeling-------- #
        self.setting_for_training()
        self.setting_for_postprocessing()
        self.setting_for_evaluation()

        self.setting_for_video_processing()

    def setting_for_preprocessing(self):
        self.setting_for_lane_representation()
        self.setting_for_svd()
        # --------others-------- #
    #系统参数
    def setting_for_system(self):
        self.gpu_id = "0" #指定哪块GPU做训练
        self.seed = 123  #固定随机种子

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        # ------------------- need to modify ------------------- #
        self.dir['head_pre'] = '/root/autodl-tmp/RVLD/RVLD-main/preprocessed/OpenLane-V' #预处理数据根目录
        # ------------------------------------------------------ #
        self.dir['pre2'] = f'{self.dir["head_pre"]}/P02_SVD/output_training/pickle' #P02处理后的数据
        self.dir['pre3_train'] = f'{self.dir["head_pre"]}/P03_video_based_datalist/output_training/pickle' #P03处理后的训练数据
        self.dir['pre3_test'] = f'{self.dir["head_pre"]}/P03_video_based_datalist/output_validation/pickle' # P03处理后的验证数据
        self.dir['model1'] = f'{os.path.dirname(os.path.dirname(self.dir["proj"]))}/ILD_seg/output' #ILD_seg模块输出/权重路径（其中seg是第一步生成掩码，coeff是第二步生成具体的车道线向量表示）
        self.dir['model2'] = f'{os.path.dirname(os.path.dirname(self.dir["proj"]))}/ILD_coeff/output' #ILD_coeff模块输出

        self.dir['out'] = f'{os.getcwd().replace("code", "output")}'
        self.dir['weight'] = f'{self.dir["out"]}/train/weight'
        self.dir['pretrained_weight1'] = f'{self.dir["model1"]}/train/weight'
        self.dir['pretrained_weight2'] = f'{self.dir["model2"]}/train/weight'
        self.dir['weight_paper'] = '/root/autodl-tmp/RVLD/RVLD-main/pretrained/OpenLane-V'

    def setting_for_dataset_path(self):
        self.dataset_name = 'openlane-v'  # ['tusimple', 'vil100']
        self.datalist = 'training'  # ['train'] only

        # ------------------- need to modify ------------------- #
        self.dir['dataset'] = '/root/autodl-tmp/RVLD/RVLD-main/OpenLane'
        # ------------------------------------------------------ #


    def setting_for_image_param(self):
        # 原始图像尺寸是 1280x1920
        self.org_height = 1280
        self.org_width = 1920
        # 网络输入尺寸是 384x640
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        # 图像归一化参数
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 480 #用于多尺度训练
        self.scale_factor = dict()
        # 网络中可能使用 多层特征融合，每一层的feature map 相对于原图的 下采样倍数8、16、32
        self.scale_factor['img'] = [8, 16, 32]
        #通常掩码的下采样较少，也就是原图 /4 例如原图 640×384/4 → 160×96
        self.scale_factor['seg'] = [4]

    def setting_for_dataloader(self):
        #4个线程
        self.num_workers = 4
        #数据增强（flip）
        self.data_flip = True
        # 将图像处理成网络可用的tensor，RVLD自定义的pipeline
        self.mode_transform = 'custom'  # ['custom', 'basic', 'complex']
        #是否按step进行采样，true从视频中按step采样帧
        self.sampling = False
        # 每个5帧采样一次
        self.sampling_step = 5
        # 按视频序列采样，如果是image就是按单帧采样
        self.sampling_mode = 'video'  # ['video', 'image']
        self.batch_size = {'img': 4} #原来是4，现在改成2
        # lane宽度设置
        self.lane_width = dict()
        self.lane_width['org'] = 5 #原始的lane宽度，也就是从原始label生成mask时lane粗细
        self.lane_width['seg'] = 1 #segmentation输出宽度，控制网络输出的lane mask粗细
        self.lane_width['mode'] = 'dilation'  # ['dilation', 'gaussian'] 膨胀卷积或者高斯模糊
        self.lane_width['sigmaX'] = 0.07 #只在mode=gaussian时生效
        self.lane_width['sigmaY'] = 0.07 #只在mode=gaussian时生效
        self.lane_width['kernel'] = (3, 3) #膨胀卷积的卷积核大小
        self.lane_width['iteration'] = 1 #mask扩张次数
        # 是否重新生成训练/测试数据列表
        self.update_datalist = True

    def setting_for_lane_representation(self):
        # 把原始 lane 点（可能稀疏、不规则）转换成 统一长度、
        # 固定 y 方向采样的点集，为后续 P02 SVD / ILD_coeff_gt 提供输入
        # 限制范围在0~270 在图像中只关注有效路面部分
        self.min_y_coord = 0
        self.max_y_coord = 270
        # 采样点的数目
        self.node_num = self.max_y_coord
        # 在区间内均匀生成node_num个y坐标点，对y坐标取整，转换到图像底部为原点的坐标系
        self.py_coord = self.height - np.float32(np.round(np.linspace(self.max_y_coord, self.min_y_coord + 1, self.node_num)))

        self.py_coord_org = np.copy(self.py_coord)
        # 插值方法，splrep平滑曲线 spline样条插值  linear线性插值 将稀疏的lane点插值到长度为node_num的向量中
        self.mode_interp = 'splrep'  # ['splrep', 'spline', 'linear', 'slinear']

    def setting_for_svd(self):
        #svd主成分数量 降维成K=64保留lane的主要形状信息
        self.top_m = 6
        # 从原始py_coord中选出索引
        # sampling lane component
        self.node_num = 100
        self.sample_idx = np.int32(np.linspace(0, self.max_y_coord - 1, self.node_num))
        # 是否启动下采样，如果是，只保留采样的100个点
        self.node_sampling = True
        if self.node_sampling == True:
            self.py_coord = self.py_coord[self.sample_idx]

    def setting_for_visualization(self):
        self.disp_step = 50
        self.disp_test_result = True

    def setting_for_save(self):
        self.save_pickle = True

    def setting_for_training(self):
        self.run_mode = 'train'  # ['train', 'test', 'eval', 'test_paper']
        # 是否从上次训练的checkpoint继续训练
        self.resume = False

        self.epochs = 50

        self.optim = dict()
        # 学习率
        self.optim['lr'] = 1e-4 #原来是1e-4
        # 正则化系数，防止过拟合
        self.optim['weight_decay'] = 1e-4
        # 衰减因子
        self.optim['gamma'] = 0.5
        # Adam动量参数
        self.optim['betas'] = (0.9, 0.999)
        # 避免除零稳定训练
        self.optim['eps'] = 1e-8
        # 选择优化器
        self.optim['mode'] = 'adam_w'  # ['adam_w', 'adam']
        #主干网络 ResNet18
        self.backbone = '18'
        # 迭代计数器
        self.iteration = dict()

    def setting_for_video_processing(self):
        # 控制模型在一个时间窗口里，用多少帧，怎么分段，什么时候开始真正启用时序学习
        # 当前帧+前1帧（利用前一帧可以稳定预测，抑制闪烁，帮助遮挡恢复）
        self.num_t = 1  # use previous {} frames
        # 视频滑动窗口长度，5帧作为一个窗口,实际输入有num_t决定
        self.window_size = 5
        #一个训练样本里取连续clip_length帧
        self.clip_length = 2
        #时序学习解锁开关，当epoch<epoch_update1只学习基础
        # epoch_update1<=epoch<epoch_update2开始引入时序
        # epoch>=epoch_update2完整视频时序学习
        self.epoch_update1 = 0
        self.epoch_update2 = 4

    def setting_for_postprocessing(self):
        # 后处理配置，网络预测完成后哪些车道线被保留，哪些被丢弃、怎样合并重要结果
        # 模型输出的不是干净的四条车道线而是一堆候选lane
        # 每条lane有概率 几何表示（ILD_coeff->还原成曲线） 位置、高度覆盖范围
        # 最多输出4条车道线
        self.max_lane_num = 4
        #
        self.pad = dict()
        # 对车道线起点和终点留白，模型预测的lane往往在边缘抖动，在最顶部和底部不稳定
        self.pad['st'] = (5, 5)  # H W
        self.pad['ed'] = (10, 10)
        # 非极大值抑制，但对象是车道线不是框
        # 两条lane如果重合度>0.5就认为是同一条车道，只保留置信度更高的一条
        self.nms_thresd = 0.5
        # 置信度阈值
        self.prob_thresd = 0.5
        # 垂直覆盖率过滤
        self.height_thresd = 0.5
        self.removal = dict()
        # 重叠车道删除
        # lane 间距 < 10px → 删除多余
        self.removal['lane_width'] = 10
    # 只保留前 4 条
    def setting_for_evaluation(self):
        # 这一段主要的作用是评估
        # 用哪个checkpoint评估
        self.param_name = 'max'  # ['trained_last', 'max']
        # 是否算LaneATT IOU指标
        self.do_eval_iou_laneatt = True
        # 是否评估时序一致性
        self.do_eval_temporal = False
        # 评估用的分辨率 原图的一半
        self.eval_h = self.org_height // 2
        self.eval_w = self.org_width // 2

        self.iou_thresd = dict()
        # IOU 判定阈值
        self.iou_thresd['laneatt'] = 0.5
        self.iou_thresd['temporal'] = 0.5
