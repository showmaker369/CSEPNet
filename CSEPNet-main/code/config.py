# -*- coding: utf-8 -*-
import os

__all__ = ["proj_root", "arg_config"]

from collections import OrderedDict

proj_root = os.path.dirname(__file__)
datasets_root = "./dataset"

SDI_tr_path = os.path.join(datasets_root, "Saliency200/TrainsetEDRNet/TR/")
SDI_te_path = os.path.join(datasets_root, "Saliency200/TrainsetEDRNet/TestSet/")


arg_config = {
    "model": "CSEPNet_VGG16",  # 实际使用的模型，需要在`network/__init__.py`中导入   CSEPNET_Res50
    "info": "",  # 关于本次实验的额外信息说明，这个会附加到本次试验的exp_name的结尾，如果为空，则不会附加内容。
    "use_amp": False,  # 是否使用amp加速训练
    "resume_mode": "",# the mode for resume parameters: ['train', 'test', '']
    "use_aux_loss": False,  # 是否使用辅助损失
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 100,  # 训练周期, 0: directly test model
    "lr": 0.001,  # 微调时缩小100倍
    "xlsx_name": "result.xlsx",  # the name of the record file
    # 数据集设置
    "rgb_data": {
        "tr_data_path": SDI_tr_path,
        "te_data_list": OrderedDict(
            {

                "TrainsetEDRNet": SDI_te_path,
            },
        ),
    },
    # 训练过程中的监控信息
    "tb_update": 0,  # >0 则使用tensorboard
    "print_freq": 50,  # >0, 保存迭代过程中的信息
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名
    "prefix": (".jpg", ".png"),
    # if you dont use the multi-scale training, you can set 'size_list': None
    # "size_list": [224, 256],
    "size_list": None,  # 不使用多尺度训练
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    # 优化器与学习率衰减
    "optim": "sgd_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,  # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 训练时是否对真值二值化（阈值为0.5）
    "batch_size": 4,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 4,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 256,
}
