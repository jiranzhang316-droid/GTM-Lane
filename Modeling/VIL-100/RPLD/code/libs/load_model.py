import torch
from models.model_s import Model as Model_S
from models.model_c import Model as Model_C
from models.model import Model
from models.loss import *

def load_model_for_test(cfg, dict_DB):
    dict_DB['model_s'] = load_pretrained_model_s(cfg)
    dict_DB['model_c'] = load_pretrained_model_c(cfg)
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_vil100_PLD')
    else:
        if cfg.param_name == 'trained_last':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        elif cfg.param_name == 'max':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_max_F1_{cfg.dataset_name}')
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['model'] = model
    return dict_DB

def load_model_for_train(cfg, dict_DB):
    model = Model(cfg=cfg)
    model.cuda()

    dict_DB['model_s'] = load_pretrained_model_s(cfg)
    dict_DB['model_c'] = load_pretrained_model_c(cfg)

    model = Model(cfg=cfg)
    model = load_for_finetuning_pretrained_model(cfg, model)
    model.cuda()

    if cfg.optim['mode'] == 'adam_w':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=cfg.optim['lr'],
                                      weight_decay=cfg.optim['weight_decay'],
                                    betas=cfg.optim['betas'], eps=cfg.optim['eps'])
    elif cfg.optim['mode'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=cfg.optim['lr'],
                                     weight_decay=cfg.optim['weight_decay'])

    cfg.optim['milestones'] = list(np.arange(0, len(dict_DB['trainloader']) * cfg.epochs, len(dict_DB['trainloader']) * 10))[1:]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.optim['milestones'],
                                                     gamma=cfg.optim['gamma'])

    if cfg.resume == False:
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.optim['milestones'],
                                                         gamma=cfg.optim['gamma'],
                                                         last_epoch=checkpoint['batch_iteration'])
        dict_DB['epoch'] = checkpoint['epoch']
        dict_DB['iteration'] = checkpoint['iteration']
        dict_DB['batch_iteration'] = checkpoint['batch_iteration']
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function(cfg)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB

def load_pretrained_model_s(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_seg_fscore_vil100_ILD_seg')

    model = Model_S(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def load_pretrained_model_c(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_vil100_ILD_coeff')
    model = Model_C(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def load_for_finetuning_pretrained_model(cfg, model):
    checkpoint1 = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}')
    checkpoint2 = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}')
    for param in list(checkpoint1['model']):
        if 'classifier' not in param:
            del checkpoint1['model'][param]

    model.load_state_dict(checkpoint1['model'], strict=False)
    # 1. 获取预训练的字典
    pretrained_dict = checkpoint2['model']
    # 2. 获取当前我们修改后模型的字典
    model_dict = model.state_dict()

    # 3. 核心修复：只保留在模型中存在，并且形状 (shape) 完全一致的权重
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.shape == model_dict[k].shape}

    # 打印一下被过滤掉的层，心里有数（可选，写论文时可以说这些层是 re-initialized 的）
    ignored_keys = set(pretrained_dict.keys()) - set(filtered_dict.keys())
    if ignored_keys:
        print(f"⚠️ 预训练权重形状不匹配，自动丢弃以下层并重新初始化: {ignored_keys}")

    # 4. 用过滤后的字典更新当前模型的字典
    model_dict.update(filtered_dict)

    # 5. 加载进模型
    model.load_state_dict(model_dict)
    model.cuda()
    return model
