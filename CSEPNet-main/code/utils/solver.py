# -*- coding: utf-8 -*-
import os
from pprint import pprint
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import network as network_lib
from loss.CEL import CEL, IOU
import loss.pytorch_ssim
from utils.dataloader import create_loader
from utils.misc import AvgMeter, construct_print, write_data_to_file
from utils.pipeline_ops import (
    make_optimizer,
    make_scheduler,
    resume_checkpoint,
    save_checkpoint,
)
from utils.recorder import TBRecorder, Timer, XLSXRecoder
from utils.py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()
device_Id = [0, 1]

class Solver:
    def __init__(self, exp_name: str, arg_dict: dict, path_dict: dict):
        super(Solver, self).__init__()
        self.exp_name = exp_name
        self.arg_dict = arg_dict
        self.path_dict = path_dict

        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.dev = torch.device(device_ids=device_Id if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()

        self.tr_data_path = self.arg_dict["rgb_data"]["tr_data_path"]
        self.te_data_list = self.arg_dict["rgb_data"]["te_data_list"]

        self.save_path = self.path_dict["save"]
        self.save_pre = self.arg_dict["save_pre"]

        if self.arg_dict["tb_update"] > 0:
            self.tb_recorder = TBRecorder(tb_path=self.path_dict["tb"])
        if self.arg_dict["xlsx_name"]:
            self.xlsx_recorder = XLSXRecoder(xlsx_path=self.path_dict["xlsx"])

        # 依赖与前面属性的属性
        self.tr_loader = create_loader(
            data_path=self.tr_data_path,
            training=True,
            size_list=self.arg_dict["size_list"],
            prefix=self.arg_dict["prefix"],
            get_length=False,
        )
        self.end_epoch = self.arg_dict["epoch_num"]
        self.iter_num = self.end_epoch * len(self.tr_loader)

        if hasattr(network_lib, self.arg_dict["model"]):
            self.net = getattr(network_lib, self.arg_dict["model"])().to(self.dev)
            # model = nn.DataParallel(getattr(network_lib, self.arg_dict["model"])(), device_ids=device_Id)
            # self.net = model.cuda(device=device_Id[0])
            # self.net = nn.DataParallel(getattr(network_lib, self.arg_dict["model"])(), device_ids=[0, 1]).to(self.dev) #mul GPU
        else:
            raise AttributeError
        pprint(self.arg_dict)

        if self.arg_dict["resume_mode"] == "test":
            # resume model only to test model.
            # self.start_epoch is useless
            resume_checkpoint(
                model=self.net,
                load_path=self.path_dict["final_state_net"],
                mode="onlynet",
            )

        # if self.arg_dict["use_aux_loss"]:
        #     self.loss_funcs.append(CEL().to(self.dev))
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=self.arg_dict["reduction"]).to(self.dev)
        self.iou = IOU(size_average=True).to(self.dev)
        self.cel = CEL().to(self.dev)
        self.ssim_loss = loss.pytorch_ssim.SSIM(window_size=11, size_average=True).to(self.dev)   # ssim_loss

        self.opti = make_optimizer(
            model=self.net,
            optimizer_type=self.arg_dict["optim"],
            optimizer_info=dict(
                lr=self.arg_dict["lr"],
                momentum=self.arg_dict["momentum"],
                weight_decay=self.arg_dict["weight_decay"],
                nesterov=self.arg_dict["nesterov"],
            ),
        )
        self.sche = make_scheduler(
            optimizer=self.opti,
            total_num=self.iter_num if self.arg_dict["sche_usebatch"] else self.end_epoch,
            scheduler_type=self.arg_dict["lr_type"],
            scheduler_info=dict(
                lr_decay=self.arg_dict["lr_decay"], warmup_epoch=self.arg_dict["warmup_epoch"]
            ),
        )

        # AMP
        if self.arg_dict["use_amp"]:
            construct_print("Now, we will use the amp to accelerate training!")
            from apex import amp

            self.amp = amp
            self.net, self.opti = self.amp.initialize(self.net, self.opti, opt_level="O1")
        else:
            self.amp = None

        if self.arg_dict["resume_mode"] == "train":
            # resume model to train the model
            self.start_epoch = resume_checkpoint(
                model=self.net,
                optimizer=self.opti,
                scheduler=self.sche,
                amp=self.amp,
                exp_name=self.exp_name,
                load_path=self.path_dict["final_full_net"],
                mode="all",
            )
        else:
            # only train a new model.
            self.start_epoch = 0
        self.sigmoid = nn.Sigmoid()

    def train(self):
        for curr_epoch in range(self.start_epoch, self.end_epoch): # 0--epoch-1
            train_loss_record = AvgMeter()
            self._train_per_epoch(curr_epoch, train_loss_record)

            # 根据周期修改学习率
            if not self.arg_dict["sche_usebatch"]:
                self.sche.step()

            # 每5个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
            if ((curr_epoch + 1) >= 85):
                save_checkpoint(
                    model=self.net,
                    optimizer=self.opti,
                    scheduler=self.sche,
                    amp=self.amp,
                    exp_name=self.exp_name,
                    current_epoch=curr_epoch + 1,
                    full_net_path=self.path_dict["final_full_net"],
                    state_net_path=self.path_dict["final_state_net"],
                )  # 保存参数
                # self.test()  # 测试每个epoch

        if self.arg_dict["use_amp"]:
            # https://github.com/NVIDIA/apex/issues/567
            with self.amp.disable_casts():
                construct_print("When evaluating, we wish to evaluate in pure fp32.")
                self.test()
        else:
            self.test()

    def cal_hybridloss(self, pre, GT):
        loss_item_list = []
        pre_sig = self.sigmoid(pre)
        CEL_loss = self.cel(pre, GT)
        BCE_loss = self.bce(pre, GT)
        Iou_loss = self.iou(pre_sig, GT)
        SSIM_loss = 1 - self.ssim_loss(pre_sig, GT)
        loss_item_list.append(f"{CEL_loss.item():.5f}")
        loss_item_list.append(f"{BCE_loss.item():.5f}")
        loss_item_list.append(f"{Iou_loss.item():.5f}")
        loss_item_list.append(f"{SSIM_loss.item():.5f}")

        Total_loss = CEL_loss + BCE_loss + Iou_loss + SSIM_loss
        return Total_loss, loss_item_list

    @Timer
    def _train_per_epoch(self, curr_epoch, train_loss_record):
        for curr_iter_in_epoch, train_data in enumerate(self.tr_loader):
            num_iter_per_epoch = len(self.tr_loader)
            curr_iter = curr_epoch * num_iter_per_epoch + curr_iter_in_epoch

            self.opti.zero_grad()
            train_inputs, train_masks, _ = train_data
            train_inputs = train_inputs.to(self.dev, non_blocking=True)
            # train_inputs = train_inputs.cuda()  #edit
            train_masks = train_masks.to(self.dev, non_blocking=True)
            # train_masks = train_masks.cuda()   #edit
            s1, s2, s3, s4, s5 = self.net(train_inputs)
            train_loss1, loss_item_list1 = self.cal_hybridloss(s1, train_masks)
            train_loss2, loss_item_list2 = self.cal_hybridloss(s2, train_masks)
            train_loss3, loss_item_list3 = self.cal_hybridloss(s3, train_masks)
            train_loss4, loss_item_list4 = self.cal_hybridloss(s4, train_masks)
            train_loss5, loss_item_list5 = self.cal_hybridloss(s5, train_masks)
            train_loss = train_loss1+train_loss2+train_loss3+train_loss4+train_loss5

            if self.amp:
                with self.amp.scale_loss(train_loss, self.opti) as scaled_loss:
                    scaled_loss.backward()
            else:
                train_loss.backward()
            self.opti.step()

            if self.arg_dict["sche_usebatch"]:
                self.sche.step()

            # 仅在累计的时候使用item()获取数据
            train_iter_loss = train_loss.item()
            train_batch_size = train_inputs.size(0)
            train_loss_record.update(train_iter_loss, train_batch_size)

            # 显示tensorboard
            if (
                self.arg_dict["tb_update"] > 0
                and (curr_iter + 1) % self.arg_dict["tb_update"] == 0
            ):
                self.tb_recorder.record_curve("trloss_avg", train_loss_record.avg, curr_iter)
                self.tb_recorder.record_curve("trloss_iter", train_iter_loss, curr_iter)
                self.tb_recorder.record_curve("lr", self.opti.param_groups, curr_iter)
                self.tb_recorder.record_image("trmasks", train_masks, curr_iter)
                self.tb_recorder.record_image("trsodout", train_preds.sigmoid(), curr_iter)
                self.tb_recorder.record_image("trsodin", train_inputs, curr_iter)
            # 记录每一次迭代的数据
            if (
                self.arg_dict["print_freq"] > 0
                and (curr_iter + 1) % self.arg_dict["print_freq"] == 0
            ):
                lr_str = ",".join(
                    [f"{param_groups['lr']:.7f}" for param_groups in self.opti.param_groups]
                )
                log = (
                    f"{curr_iter_in_epoch}:{num_iter_per_epoch}/"
                    f"{curr_iter}:{self.iter_num}/"
                    f"{curr_epoch}:{self.end_epoch} "
                    f"{self.exp_name}\n"
                    f"Lr:{lr_str} "
                    f"M:{train_loss_record.avg:.5f} C:{train_iter_loss:.5f} "
                    f"{loss_item_list5}"
                )
                print(log)
                write_data_to_file(log, self.path_dict["tr_log"])

    def test(self):
        self.net.eval()
        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader = create_loader(
                data_path=data_path,
                training=False,
                size_list=None,
                prefix=self.arg_dict["prefix"],
                get_length=False,
            )
            self.save_path = os.path.join(self.path_dict["save"], data_name)
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self._test_process(save_pre=self.save_pre)
            msg = f"Results on the testset({data_name}:'{data_path}'): {results}"
            construct_print(msg)
            write_data_to_file(msg, self.path_dict["te_log"])

            total_results[data_name] = results

        self.net.train()

        if self.arg_dict["xlsx_name"]:
            # save result into xlsx file.
            self.xlsx_recorder.write_xlsx(self.exp_name, total_results)

    def _test_process(self, save_pre):
        loader = self.te_loader

        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{test_batch_id + 1}")
            with torch.no_grad():
                in_imgs, in_mask_paths, in_names = test_data
                in_imgs = in_imgs.to(self.dev, non_blocking=True)
                outputs, s2, s3, s4, s5 = self.net(in_imgs)  #output就是s1 修改后网络输出有5个
            outputs_np = outputs.sigmoid().cpu().detach()

            for item_id, out_item in enumerate(outputs_np):
                gimg_path = os.path.join(in_mask_paths[item_id])
                gt_img = Image.open(gimg_path).convert("L")
                out_img = self.to_pil(out_item).resize(gt_img.size, resample=Image.NEAREST)

                if save_pre:
                    oimg_path = os.path.join(self.save_path, in_names[item_id] + ".png")
                    out_img.save(oimg_path)

                gt_img = np.array(gt_img)
                out_img = np.array(out_img)
                FM.step(pred=out_img, gt=gt_img)
                WFM.step(pred=out_img, gt=gt_img)
                SM.step(pred=out_img, gt=gt_img)
                EM.step(pred=out_img, gt=gt_img)
                MAE.step(pred=out_img, gt=gt_img)
        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]
        results = {
            "Smeasure": sm,
            # "wFmeasure": wfm,
            "MAE": mae,
            # "adpEm": em["adp"],
            # "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            # "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }
        return results
