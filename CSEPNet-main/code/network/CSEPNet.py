# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from backbone.origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
from module.BaseBlocks import BasicConv2d
from module.MyModule import *
from utils.tensor_ops import cus_sample

#VGG-16 backbone
class CSEPNet_VGG16(nn.Module):
    def __init__(self):
        super(CSEPNet_VGG16, self).__init__()

        self.upsample = cus_sample

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()


        self.Gate1 = CBAM(64)
        self.Gate2 = CBAM(128)
        self.Gate3 = CBAM(256)
        self.Gate4 = CBAM(512)
        self.Gate5 = CBAM(512)


        #1*1 conv change channel
        self.con_AIM5 = nn.Conv2d(512, 64, 3, 1, 1)
        self.con_AIM4 = nn.Conv2d(512, 64, 3, 1, 1)
        self.con_AIM3 = nn.Conv2d(256, 64, 3, 1, 1)
        self.con_AIM2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.con_AIM1 = nn.Conv2d(64, 32, 3, 1, 1)

        # CSC Module
        self.CSC16 = CSC(64)
        self.CSC8 = CSC(64)
        self.CSC4 = CSC(64)
        self.CSC2 = CSC(32)

        # CSFI Module
        self.sim16 = CSFI(64, 32)
        self.sim8 = CSFI(64, 32)
        self.sim4 = CSFI(64, 32)
        self.sim2 = CSFI(64, 32)
        self.sim1 = CSFI(32, 16)


        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)
        # decoder---用来监督
        self.decoder = decoder()
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, in_data):
            in_data_1 = self.encoder1(in_data)
            in_data_2 = self.encoder2(in_data_1)
            in_data_4 = self.encoder4(in_data_2)
            in_data_8 = self.encoder8(in_data_4)
            in_data_16 = self.encoder16(in_data_8)

            in_data_1 = self.Gate1(in_data_1)
            in_data_2 = self.Gate2(in_data_2)
            in_data_4 = self.Gate3(in_data_4)
            in_data_8 = self.Gate4(in_data_8)
            in_data_16 = self.Gate5(in_data_16)

        #1*1 CONV CHANGE CHANNEL
            in_data_1 = self.con_AIM1(in_data_1)
            in_data_2 = self.con_AIM2(in_data_2)
            in_data_4 = self.con_AIM3(in_data_4)
            in_data_8 = self.con_AIM4(in_data_8)
            in_data_16 = self.con_AIM5(in_data_16)

      # SIM_ x -- 1*1 CONV
            p5 = self.sim16(in_data_16)
            out_data_16 = self.upconv16(self.sim16(in_data_16))


            out_data_8 = self.CSC16(in_data_8,out_data_16)
            p4 = self.sim8(out_data_8)
            out_data_8 = self.upconv8(self.sim8(out_data_8))  # 512


            out_data_4 = self.CSC8(in_data_4,out_data_8)
            p3 = self.sim4(out_data_4)
            out_data_4 = self.upconv4(self.sim4(out_data_4))  # 256


            out_data_2 = self.CSC4(in_data_2,out_data_4)
            p2 = self.sim2(out_data_2)
            out_data_2 = self.upconv2(self.sim2(out_data_2))  # 64


            out_data_1 = self.CSC2(in_data_1,out_data_2)

            out_data_1 = self.upconv1(self.sim1(out_data_1))  # 32


            out_data = self.classifier(out_data_1)
            p1 = out_data
            s1, s2, s3, s4, s5 = self.decoder(p1, p2, p3, p4, p5)
            s5 = self.upsample8(s5)
            s4 = self.upsample4(s4)
            s3 = self.upsample2(s3)
            return s1, s2, s3, s4, s5


#ResNet-50 backbone
class CSEPNET_Res50(nn.Module):
    def __init__(self):
        super(CSEPNET_Res50, self).__init__()
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()

        self.upsample = cus_sample

        self.Gate1 = CBAM(64)  #11.20 NIGHT
        self.Gate2 = CBAM(256)
        self.Gate3 = CBAM(512)
        self.Gate4 = CBAM(1024)
        self.Gate5 = CBAM(2048)


        #1*1 conv change channel
        self.con_AIM5 = nn.Conv2d(2048, 64, 3, 1, 1)
        self.con_AIM4 = nn.Conv2d(1024, 64, 3, 1, 1)
        self.con_AIM3 = nn.Conv2d(512, 64, 3, 1, 1)
        self.con_AIM2 = nn.Conv2d(256, 64, 3, 1, 1)
        self.con_AIM1 = nn.Conv2d(64, 32, 3, 1, 1)

        # 自定义CSC模块
        self.CSC16 = CSC(64)
        self.CSC8 = CSC(64)
        self.CSC4 = CSC(64)
        self.CSC2 = CSC(32)

        # SIM_1模块
        self.sim16 = CSFI(64, 32)
        self.sim8 = CSFI(64, 32)
        self.sim4 = CSFI(64, 32)
        self.sim2 = CSFI(64, 32)
        self.sim1 = CSFI(32, 16)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)
        # decoder---用来监督
        self.decoder = decoder()
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, in_data):
        in_data_2 = self.div_2(in_data)
        in_data_4 = self.div_4(in_data_2)
        in_data_8 = self.div_8(in_data_4)
        in_data_16 = self.div_16(in_data_8)
        in_data_32 = self.div_32(in_data_16)


        in_data_1 = self.Gate1(in_data_2)  #CBAM
        in_data_2 = self.Gate2(in_data_4)
        in_data_4 = self.Gate3(in_data_8)
        in_data_8 = self.Gate4(in_data_16)
        in_data_16 = self.Gate5(in_data_32)

        #1*1 CONV CHANGE CHANNEL
        in_data_1 = self.con_AIM1(in_data_1)
        # print('in_data_16size {} '.format(in_data_1.shape)) ([4, 32, 128, 128])
        in_data_2 = self.con_AIM2(in_data_2)
        # print('in_data_16size {} '.format(in_data_2.shape))  ([4, 64, 64, 64])
        in_data_4 = self.con_AIM3(in_data_4)
        # print('in_data_16size {} '.format(in_data_4.shape))  ([4, 64, 32, 32])
        in_data_8 = self.con_AIM4(in_data_8)
        # print('in_data_16size {} '.format(in_data_8.shape))  ([4, 64, 16, 16])
        in_data_16 = self.con_AIM5(in_data_16)
        # print('in_data_16size {} '.format(in_data_16.shape))  ([4, 64, 8, 8])

      # SIM_ x -- 1*1 CONV
        p5 = self.sim16(in_data_16)
        out_data_16 = self.upconv16(self.sim16(in_data_16))

        out_data_8 = self.CSC16(in_data_8,out_data_16)
        p4 = self.sim8(out_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8))  # 512

        out_data_4 = self.CSC8(in_data_4,out_data_8)
        p3 = self.sim4(out_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4))  # 256

        out_data_2 = self.CSC4(in_data_2,out_data_4)
        p2 = self.sim2(out_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2))  # 64

        out_data_1 = self.CSC2(in_data_1,out_data_2)

        out_data_1 = self.upconv1(self.sim1(out_data_1))  # 32

        out_data = self.classifier(out_data_1)
        p1 = out_data
        s1, s2, s3, s4, s5 = self.decoder(p1, p2, p3, p4, p5)
        s5 = self.upsample16(s5)
        s4 = self.upsample8(s4)
        s3 = self.upsample4(s3)
        s2 = self.upsample2(s2)
        s1 = self.upsample2(s1)
        return s1, s2, s3, s4, s5



if __name__ == "__main__":
    net = CSEPNet_VGG16()
    y = net(input)