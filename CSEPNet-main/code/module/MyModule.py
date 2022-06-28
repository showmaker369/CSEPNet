import torch
import torch.nn as nn
from utils.tensor_ops import cus_sample

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.decoder5 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S5 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder4 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S4 = nn.Conv2d(64, 1, 3, stride=1, padding=1)


        self.decoder3 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        # 64 Channel
        self.decoder2 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.S1 = nn.Conv2d(1, 1, 3, stride=1, padding=1)


    def forward(self, x1, x2, x3, x4, x5):

        x5_up = self.decoder5(x5)
        s5 = self.S5(x5_up)
        # print('x5_up size {} '.format(x5_up.shape))

        x4_up = self.decoder4(x4)
        s4 = self.S4(x4_up)
        # print('x4_up size {} '.format(x4_up.shape))

        x3_up = self.decoder3(x3)
        s3 = self.S3(x3_up)
        # print('x3_up size {} '.format(x3_up.shape))

        x2_up = self.decoder2(x2)
        s2 = self.S2(x2_up)
        # print('x2_up size {} '.format(x2_up.shape))

        s1 = self.S1(x1)
        # print('s1 size {} '.format(s1.shape))

        return s1, s2, s3, s4, s5


class CSC(nn.Module):
    def __init__(self, left_channel):
        super(CSC, self).__init__()
        self.upsample = cus_sample
        self.conv1 = BasicConv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.conv_cat = BasicConv2d(2*left_channel, left_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fusion = BasicConv2d(left_channel, left_channel, 3, padding=1)

    def forward(self, left, right):
        right = self.upsample(right, scale_factor=2)  # right 上采样
        right = self.conv1(right)
        x1 = left.mul(self.sa2(right))+left
        x2 = right.mul(self.sa1(left))+right
        mid = self.conv_cat(torch.cat((x1, x2), 1)) #concat
        mid = self.conv2(mid)
        mid = self.sigmoid(mid)
        out = left * mid + right * mid
        out = self.fusion(out)

        return out




class CSWeight(nn.Module):
    def __init__(self, left_channel, right_channel):
        super(CSWeight, self).__init__()
        self.upsample = cus_sample
        self.conv1 = BasicConv2d(left_channel, right_channel, kernel_size=3, stride=1, padding=1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.conv_cat = BasicConv2d(2*right_channel, right_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(right_channel, right_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fusion = BasicConv2d(right_channel, right_channel, 3, padding=1)

    def forward(self, left, right):
        left = self.upsample(left, scale_factor=2)  # right 上采样
        left = self.conv1(left)
        x1 = left.mul(self.sa2(right))+left
        x2 = right.mul(self.sa1(left))+right
        mid = self.conv_cat(torch.cat((x1, x2), 1)) #concat
        mid = self.conv2(mid)
        wei = self.sigmoid(mid)
        out = self.fusion(left * wei + right * (1 - wei))  # wei*left + (1-wei)*right
        return out



#CBAM
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        CA = x.mul(self.ca(x))
        SA = CA.mul(self.sa(CA))
        return SA

class MSCA(nn.Module):
    def __init__(self, channels, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        # xg = self.global_att(x)
        xg=x
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Contrast(nn.Module):
    def __init__(self, in_c):
        super(Contrast, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1,padding=1)
        self.conv_1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = x-self.avg_pool(x)  #Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight*x + x

        return out#Res


#Module-CSFI
class CSFI(nn.Module):
    def __init__(self, h_C, l_C):
        super(CSFI, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.h2h_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample
        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.h1h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.contrast1 = Contrast(l_C) #Contrast模块
        self.contrast2 = Contrast(l_C)  # Contrast模块
        self.contrast3 = Contrast(h_C) #Contrast模块
        self.contrast4 = Contrast(h_C)  # Contrast模块

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_2 = nn.Conv2d(l_C, l_C, 3, 1, 1)

        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.csweight = CSWeight(l_C, h_C)
        self.relu = nn.ReLU(True)
        self.conv_out = nn.Conv2d(h_C, h_C, 3, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.contrast3(x_h)# 经过contrast层
        x_h2h = self.h2h_1(x_h2h)
        x_l2l = self.contrast1(x_l)# 经过contrast层
        x_l2l = self.l2l_1(x_l2l)
        x_l2h = self.l2h_1(self.l2h_up(x_l2l, size=(h, w)))
        x_h = self.relu(self.bnh_1(self.h1h_1(x_h2h + x_l2h)))
        x_h = self.contrast4(x_h)# 经过contrast层
        x_h2l = self.h2h_pool(x_h2h)
        x_h2l = self.h2l_1(x_h2l)
        x_l = self.relu(self.bnl_1(self.l2l_0(x_l2l+x_h2l)))
        x_l = self.contrast2(x_l)# 经过contrast层

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l = self.l2l_2(x_l)
        x_h = self.csweight(x_l, x_h2h)
        out = self.conv_out(x_h + x)
        return out


class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



if __name__ == "__main__":
    module = CSFI(64, 32)
    # print([(name, params.size()) for name, params in module.named_parameters()])
