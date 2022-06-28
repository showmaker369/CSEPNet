# **CSEPNet**

This project include the code and results for 'Cross-Scale Edge Purification Network for Salient Object Detection
of Steel Defect Images', Measurement, accepted, 2022.   [**paper links**:](https://doi.org/10.1016/j.measurement.2022.111429)

## Network Architecture

   <div align=center>
   <img src="https://github.com/showmaker369/CSEPNet/blob/main/network.PNG">
   </div>

## Our Environments 

Ubuntu                             16.04
matplotlib                         3.2.2
numpy                              1.19.2
pip                                21.2.2
timm                               0.5.4
torch                              1.1.0
torchstat                          0.0.6
torchvision                        0.3.0

## Saliency maps & dataset

 We provide saliency maps of our CSEPNet ([VGG_backbone](https://pan.baidu.com/s/1vTg5Zqs3G4RMiElVk22shg) (code: 8gde )on SD-saliency-900 dataset.

You can also download [dataset of our model](https://pan.baidu.com/s/1Y3fObBrnbpWkSEf3LvWXSA) (code: hlpi)) and [Pretrained Parameters](https://pan.baidu.com/s/1CajYCSMmy7pfteD4jRrkig ) (code: gxc4 ).


## Folders & Files

<details>
<summary>Directory Structure</summary>

```shell script
$ tree -L 3
.
├── backbone
│   ├── __init__.py
│   ├── origin
│   │   ├── from_origin.py
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   └── wsgn
│       ├── customized_func.py
│       ├── from_wsgn.py
│       ├── __init__.py
│       ├── resnet.py
│       └── resnext.py
├── config.py
├── LICENSE
├── loss
│   ├── CEL.py,pytorch_ssim
│   └── __init__.py
├── main.py
├── module
│   ├── BaseBlocks.py
│   ├── __init__.py
│   ├── MyModule.py

├── network
│   ├── __init__.py
│   ├── CSEPNet.py
├── output (These are the files generated when I ran the code.)
│   ├── result
│   │   ├── cfg_2020-07-23.txt
│   │   ├── pre
│   │   ├── pth(generate in training process)
│   └── result.xlsx   
│         
├── readme.md
└── utils
    ├── py_sod_metrics(code to calculate metrics)
    ├── dataloader.py
    ├── __init__.py
    ├── joint_transforms.py
    ├── metric.py
    ├── misc.py
    ├── pipeline_ops.py
    ├── recorder.py
    ├── solver.py
    └── tensor_ops.py

```
</details>

* `backbone`: Store some code for backbone networks.
* `loss`: The code of the loss function.
* `module`: The code of important modules.
* `network`: The code of the network.
* `output`: It saves all results.
* `utils`: Some instrumental code.
    * `dataloader.py`: About creating the dataloader.
    * ...
* `main.py`: I think you can understand.
* `config.py`: Configuration file for model training and testing.



```python
# config.py
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
        "tr_data_path": SDI_tr_path,  ##(训练集path)
        "te_data_list": OrderedDict(
            {

                "TrainsetEDRNet": SDI_te_path, ##(测试集path)
            },
        ),
    },
    # 训练过程中的监控信息
    "tb_update": 0,  # >0 则使用tensorboard
    "print_freq": 50,  # >0, 保存迭代过程中的信息
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名
    "prefix": (".jpg", ".png"),
    # if you dont use the multi-scale training, you can set 'size_list': None
    # "size_list": [224, 256, 288, 320, 352],
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
```

</details>

## Train

1. You can customize the value of the [`arg_config`](config.py#L20) dictionary in the configuration file.
    * The first time you use it, you need to adjust the [path](config.py#L9-L17) of every dataset.
    * Set `model` to the model name that has been imported in `network/__init__.py`.
    * Modify `info` according to your needs, which will be appended to the final `exp_name`. (The function `construct_exp_name` in `utils/misc.py` will generate the final `exp_name`.)
    * set the "resume_mode" in 【arg_config】 of  "config.py" as " "( if set it as "train", the model will continue to train from the saved pth)
    * And other setting in `config.py`, like `epoch_num`, `lr` and so on...
2. In the folder `code`, run the command `python main.py`.
3. Everything is OK. Just wait for the results.
4. The test will be performed automatically when the training is completed.
5. All results will be saved into the folder `output`, including predictions in folder `pre` (if you set `save_pre` to `True`), `.pth` files in folder `pth` and other log files.

## If you want to **test** the trained model again...

**Our pre-training parameters can also be used in this way.**

1. In the `output` folder, please ensure that there is **a folder corresponding to the model (See [Note](#Note))**, which contains the `pth` folder, and the `.pth` file of the model is located here and its name is `state_final.pth`.
2. Set the value of `model` of `arg_config` to the model you want to test.
3. Set the value of `te_data_list` to your dataset path.
4. Set the value of `resume_mode` to `test`.
5. In the folder `code`, run `python main.py`.
6. You can find predictions from the model in the folder `pre` of the `output`.

## Evaluation

- You can use the [evaluation tool (MATLAB version)](https://github.com/DengPingFan/CODToolbox) to evaluate the above saliency maps.
- Thanks for the code (python version) provided by  [Lartpang](https://github.com/lartpang/PySODMetrics) , we use the code to evaluate the metrics for testing
- We also give thanks to Fan for the matlab code (https://github.com/DengPingFan/CODToolbox)

## Thanks

Our code is based on **MINet**: https://github.com/lartpang/MINet, thanks for the authors!!



## Reference

@inproceedings{fan2017structure,
	title={{Structure-measure: A New Way to Evaluate Foreground Maps}},
	author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
	booktitle={IEEE International Conference on Computer Vision (ICCV)},
	pages = {4548-4557},
	year={2017},
	note={\url{http://dpfan.net/smeasure/}},
	organization={IEEE}
}

@inproceedings{Fan2018Enhanced,
	author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
	title={{Enhanced-alignment Measure for Binary Foreground Map Evaluation}},
	booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
	pages={698--704},
	note={\url{http://dpfan.net/e-measure/}},
	year={2018}
}

@inproceedings{MINet-CVPR2020,
    author = {Pang, Youwei and Zhao, Xiaoqi and Zhang, Lihe and Lu, Huchuan},
    title = {Multi-Scale Interactive Network for Salient Object Detection},
    booktitle = CVPR,
    month = {June},
    year = {2020}
}

## Note

For any questions, please contact me at tding97@163.com.

</details>
