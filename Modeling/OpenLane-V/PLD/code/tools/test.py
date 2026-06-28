import numpy as np
from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.post_process = dict_DB['post_process']
        self.save_pred_for_eval_iou = dict_DB['save_pred_for_eval_iou']
        self.eval_iou_laneatt = dict_DB['eval_iou_laneatt']
        self.eval_seg = dict_DB['eval_seg']
        self.eval_flow = dict_DB['eval_flow']
        self.eval_temporal = dict_DB['eval_temporal']
        self.visualizer = dict_DB['visualizer']

        self.vm = dict_DB['video_memory']

    def init_data(self):
        self.result = {'out': {}, 'gt': {}, 'name': []}
        self.datalist = []
        self.eval_seg.init()

    def batch_to_cuda(self, batch):
        for name in list(batch):
            if torch.is_tensor(batch[name]):
                batch[name] = batch[name].cuda()
            elif type(batch[name]) is dict:
                for key in batch[name].keys():
                    batch[name][key] = batch[name][key].cuda()
        return batch

    def run(self, model_s, model_c, model, mode='val'):
        self.init_data()

        with torch.no_grad():
            model_s.eval()
            model_c.eval()
            model.eval()

            for i, batch in enumerate(self.testloader):  # load batch data
                batch = self.batch_to_cuda(batch)

                # model
                out = dict()
                model_s.forward_for_encoding(batch['img']) #提取图片特征
                model_s.forward_for_squeeze()  #对特征进行聚合
                out.update(model_s.forward_for_classification()) #分类头产生初步的 seg_map
                model_c.prob_map = model_s.prob_map[:, 1:]
                out.update(model_c.forward_for_regression())  #回归头产生初步的coeff (回归参数)
                for j in range(len(batch['prev_num'])):
                    prev_frame_num = int(batch['prev_num'][j])

                    # model 记忆容器管理
                    key_t = f't-0'
                    if prev_frame_num == 0:
                        self.vm.forward_for_dict_initialization() # 如果是视频第一帧，清空记忆
                    else:
                        self.vm.forward_for_dict_memorization() # 否则，继承之前的记忆

                    model.clip_idx = prev_frame_num
                    # 把当前单帧模型的特征存入 VM，位置是 't-0'
                    self.vm.forward_for_dict_initialization_per_frame(key_t)
                    self.vm.forward_for_dict_update_per_frame(model_s, batch_idx=j, mode='intra')

                    if prev_frame_num >= self.cfg.num_t:
                        # 只有当历史帧积累足够多 (例如 num_t=1)，才启动视频模型
                        model = self.vm.forward_for_dict_transfer(model)  # 把 VM 里的 t-0, t-1 灌入 model
                        out.update(model.forward_for_feat_aggregation()) #进行光流对齐得到对齐后的特征
                        out.update(model.forward_for_classification()) #进行分类
                        out.update(model.forward_for_regression()) #进行回归
                        # 把融合后的特征再存回 VM，供下一帧参考
                        self.vm.forward_for_dict_update_per_frame(model, mode='update')

                    # lane mask guide
                    self.post_process.mode = ('f' if prev_frame_num >= self.cfg.num_t else 'init')
                    out_post = self.post_process.run_for_test(out, batch_idx=j)
                    out.update(self.post_process.lane_mask_generation(out_post))
                    self.vm.data['guide_cls']['t-0'] = out['guide_cls']

                    self.eval_seg.update(batch, out, batch_idx=j, prev_frame_num=prev_frame_num, mode=mode)
                    self.eval_seg.run_for_fscore()

                    # visualize
                    if self.cfg.disp_test_result == True and ('train' not in mode):
                        out.update(out_post[0])
                        self.visualizer.display_for_test(batch=batch, out=out, prev_frame_num=prev_frame_num, batch_idx=j, mode=mode)

                    # record output data
                    self.result['out']['x_coords'] = out_post[0]['x_coords']
                    self.result['out']['height_idx'] = out_post[0]['height_idx']
                    self.result['name'] = batch['img_name'][j]

                    if self.cfg.save_pickle == True:
                        save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/{batch["img_name"][j].replace(".jpg", "")}', data=self.result)

                    self.datalist.append(batch['img_name'][j])

                if i % 50 == 1:
                    print(f'image {i} ---> {batch["img_name"][0]} done!')

        if self.cfg.save_pickle == True:
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/datalist', data=self.datalist)
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/eval_seg_results', data=self.eval_seg.results)

        # evaluation
        return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()
        try:
            metric.update(self.eval_seg.measure())
        except:
            print('evaluation mode!')

        if self.cfg.do_eval_iou_laneatt == True:
            self.save_pred_for_eval_iou.settings(key=['x_coords'], test_mode=mode, use_height=True)
            self.save_pred_for_eval_iou.run()
            metric.update(self.eval_iou_laneatt.measure_IoU(mode, self.cfg.iou_thresd['laneatt']))

        if self.cfg.do_eval_temporal == True and self.cfg.run_mode == 'test_paper':
            metric.update(self.eval_temporal.measure_IoU(mode, self.cfg.iou_thresd['temporal']))


        return metric
