import os

from options.config import Config
from options.args import *
from tools.test import *
from tools.train import *
from libs.prepare import *
def main_eval(cfg, dict_DB):
    # eval option
    test_process = Test_Process(cfg, dict_DB)
    test_process.evaluation(mode='test')

def main_test(cfg, dict_DB):
    # test option
    test_process = Test_Process(cfg, dict_DB)
    test_process.run(dict_DB['model_s'], dict_DB['model_c'], dict_DB['model'], mode='test')

def main_train(cfg, dict_DB):
    # train option
    dict_DB['test_process'] = Test_Process(cfg, dict_DB)
    # train_process = new_Train_Process(cfg, dict_DB)
    train_process = Train_Process(cfg, dict_DB)
    train_process.run()

def main():
    # Config
    cfg = Config()
    cfg = parse_args(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    # 加载可视化 读出U形状（100，6）
    dict_DB = prepare_visualization(cfg , dict_DB)
    # 加载数据集 训练集的数量为16567* batch_size 4 测试集的数量为 batch_size 1439*16
    dict_DB = prepare_dataloader(cfg, dict_DB)
    # 加载模型
    dict_DB = prepare_model(cfg, dict_DB)
    # 加载预处理
    dict_DB = prepare_post_processing(cfg, dict_DB)
    dict_DB = prepare_evaluation(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)

    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    elif 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    elif 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)


if __name__ == '__main__':
    main()
