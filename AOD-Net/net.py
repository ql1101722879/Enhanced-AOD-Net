import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


import matplotlib.pyplot as plt
from torch.nn import init
import math


# class dehaze_net(nn.Module):
#
# 	def __init__(self):
# 		super(dehaze_net, self).__init__()
#
# 		self.relu = nn.ReLU(inplace=True)
#
# 		self.e_conv1 = nn.Conv2d(3,3,1,1,0,bias=True)
# 		self.e_conv2 = nn.Conv2d(3,3,3,1,1,bias=True)
# 		self.e_conv3 = nn.Conv2d(6,3,5,1,2,bias=True)
# 		self.e_conv4 = nn.Conv2d(6,3,7,1,3,bias=True)
# 		self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True)
#
# 	def forward(self, x):
# 		source = []
# 		source.append(x)
#
# 		x1 = self.relu(self.e_conv1(x))
# 		x2 = self.relu(self.e_conv2(x1))
#
# 		concat1 = torch.cat((x1,x2), 1)
# 		x3 = self.relu(self.e_conv3(concat1))
#
# 		concat2 = torch.cat((x2, x3), 1)
# 		x4 = self.relu(self.e_conv4(concat2))
#
# 		concat3 = torch.cat((x1,x2,x3,x4),1)
# 		x5 = self.relu(self.e_conv5(concat3))
#
# 		clean_image = self.relu((x5 * x) - x5 + 1)
#
# 		return clean_image


class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)


## 1.AOD-Net改进网络：1：混合空洞卷积
class AODnet_ours1(nn.Module):
    def __init__(self):
        super(AODnet_ours1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv3_1 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv3_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=6, dilation=3)
        self.conv4_1 = nn.Conv2d(in_channels=36, out_channels=3, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv4_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=6, dilation=2)
        self.conv4_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=9, dilation=3)
        self.conv5 = nn.Conv2d(in_channels=114, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1
    #
    # def visualize_and_save(self, tensor, name):
    #     # 可视化并保存张量为图像
    #     tensor = tensor[:, :3, :, :]
    #     grid_img = make_grid(tensor, nrow=4, padding=1, normalize=True)
    #     img = to_pil_image(grid_img)
    #     img.save(f"{name}.png")



    def forward(self, x):
        # print(x.size())
        x1 = F.relu(self.conv1(x))  # 3
        # print(x1.size())   # 3 640 480
        x2_1 = F.relu(self.conv2_1(x1))  # 3
        # print(x2_1.size()) # 3 640 480
        x2_2 = F.relu(self.conv2_2(x2_1))
        # print(x2_2.size()) # 3 640 480
        x2_3 = F.relu(self.conv2_3(x2_2))
        # print(x2_3.size()) # 3 640 480
        x2_4 = torch.cat((x1, x2_3), 1)

        x2 = torch.cat((x1, x2_4), 1)
        # print(x2.size()) # torch.Size([2, 9, 640, 480])
        cat1 = torch.cat((x1, x2), 1)  # 6

        # print(cat1.size()) # torch.Size([2, 12, 640, 480])
        x3_1 = F.relu(self.conv3_1(cat1))  # 3
        # print(x3_1.size()) # torch.Size([2, 3, 640, 480])
        x3_2 = F.relu(self.conv3_2(x3_1))
        # print(x3_2.size()) # torch.Size([2, 3, 640, 480])
        x3_3 = F.relu(self.conv3_3(x3_2))
        # print(x3_3.size()) # torch.Size([2, 3, 640, 480])
        x3_4 = torch.cat((cat1, x3_3), 1)

        x3 = torch.cat((cat1, x3_4), 1)
        # prin t(x3.size())  # torch.Size([2, 27, 640, 480])

        cat2 = torch.cat((x2, x3), 1)  # 9
        # print(cat2.size()) # torch.Size([2, 36, 640, 480])
        x4_1 = F.relu(self.conv4_1(cat2))  # 3
        # print(x4_1.size()) # 3 640 480
        x4_2 = F.relu(self.conv4_2(x4_1))
        # print(x4_2.size()) # 3 640 480
        x4_3 = F.relu(self.conv4_3(x4_2))
        # print(x4_3.size()) # 3 640 480
        x4_4 = torch.cat((cat2, x4_3), 1)
        # print(x4_4.size())

        x4 = torch.cat((cat2, x4_4), 1)
        # print(x4.size()) # torch.Size([2, 81, 640, 480])
        cat3 = torch.cat((x1, x2, x3, x4), 1)  # 12


        # print(cat3.size()) # torch.Size([2, 120, 640, 480])
        # print(cat3.size())  # 45 640 480
        # x5_1 = self.eca_5(cat3)
        # x5_2 = self.pa_5(x5_1)
        k = F.relu(self.conv5(cat3))


        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)







## AOD-Net改进网络：1：混合空洞卷积 2 ECA注意力机制 3 PA注意力机制 多尺度特征融合增强 FFA设及思想。
class AODnet_ours(nn.Module):
    def __init__(self):
        super(AODnet_ours, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv3_1 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv3_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=6, dilation=3)
        self.conv4_1 = nn.Conv2d(in_channels=36, out_channels=3, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv4_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=6, dilation=2)
        self.conv4_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=9, dilation=3)
        self.conv5 = nn.Conv2d(in_channels=114, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.eca_2 = eca_layer(channel=6, k_size=3)
        self.eca_3 = eca_layer(channel=15, k_size=5)
        self.eca_4 = eca_layer(channel=39, k_size=7)
        # self.eca_5 = eca_layer(channel=120, k_size=3)
        self.pa_2 = PA(nf=6)
        self.pa_3 = PA(nf=15)
        self.pa_4 = PA(nf=39)
        # self.pa_5 = PA(nf=120)
        self.b = 1

    def forward(self, x):
        # print(x.size())
        x1 = F.relu(self.conv1(x))   # 3
        # print(x1.size())   # 3 640 480
        x2_1 = F.relu(self.conv2_1(x1))  # 3
        # print(x2_1.size()) # 3 640 480
        x2_2 = F.relu(self.conv2_2(x2_1))
        # print(x2_2.size()) # 3 640 480
        x2_3 = F.relu(self.conv2_3(x2_2))
        # print(x2_3.size()) # 3 640 480
        x2_4 = torch.cat((x1, x2_3), 1)
        # print(x2_4.size()) # torch.Size([2, 6, 640, 480])
        x2_5 = self.eca_2(x2_4)
        # print(x2_5.size()) # torch.Size([2, 6, 640, 480])
        x2_6 = self.pa_2(x2_5)
        # print(x2_6.size()) # torch.Size([2, 6, 640, 480])
        x2 = torch.cat((x1,x2_6), 1)
        # print(x2.size()) # torch.Size([2, 9, 640, 480])
        cat1 = torch.cat((x1, x2), 1) # 6
        # print(cat1.size()) # torch.Size([2, 12, 640, 480])
        x3_1 = F.relu(self.conv3_1(cat1)) # 3
        # print(x3_1.size()) # torch.Size([2, 3, 640, 480])
        x3_2 = F.relu(self.conv3_2(x3_1))
        # print(x3_2.size()) # torch.Size([2, 3, 640, 480])
        x3_3 = F.relu(self.conv3_3(x3_2))
        # print(x3_3.size()) # torch.Size([2, 3, 640, 480])
        x3_4 = torch.cat((cat1,x3_3), 1)
        # print(x3_4.size()) # torch.Size([2, 15, 640, 480])
        x3_5 = self.eca_3(x3_4)
        # print(x3_5.size()) # torch.Size([2, 15, 640, 480])
        x3_6 = self.pa_3(x3_5)
        # print(x3_6.size()) # torch.Size([2, 15, 640, 480])
        x3 = torch.cat((cat1, x3_6), 1)
        # prin t(x3.size())  # torch.Size([2, 27, 640, 480])
        cat2 = torch.cat((x2, x3), 1) # 9
        # print(cat2.size()) # torch.Size([2, 36, 640, 480])
        x4_1 = F.relu(self.conv4_1(cat2)) # 3
        # print(x4_1.size()) # 3 640 480
        x4_2 = F.relu(self.conv4_2(x4_1))
        # print(x4_2.size()) # 3 640 480
        x4_3 = F.relu(self.conv4_3(x4_2))
        # print(x4_3.size()) # 3 640 480
        x4_4 = torch.cat((cat2, x4_3), 1)
        # print(x4_4.size())
        x4_5 = self.eca_4(x4_4)
        # print(x4_5.size())
        x4_6 = self.pa_4(x4_5)
        # print(x4_6.size())
        x4 = torch.cat((cat2, x4_6), 1)
        # print(x4.size()) # torch.Size([2, 81, 640, 480])
        cat3 = torch.cat((x1, x2, x3, x4), 1) # 12
        # print(cat3.size()) # torch.Size([2, 120, 640, 480])
        # print(cat3.size())  # 45 640 480
        # x5_1 = self.eca_5(cat3)
        # x5_2 = self.pa_5(x5_1)
        k = F.relu(self.conv5(cat3))
        return k

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out