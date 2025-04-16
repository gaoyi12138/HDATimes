import torch
from torch import nn
from layers.danet import DANetHead


class PSDAModule(nn.Module):
    def __init__(self, inplans=64, planes=64, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSDAModule, self).__init__()
        self.conv_1 = nn.Conv2d(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                stride=stride, groups=conv_groups[0])
        self.conv_2 = nn.Conv2d(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                stride=stride, groups=conv_groups[1])
        self.conv_3 = nn.Conv2d(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                stride=stride, groups=conv_groups[2])
        self.conv_4 = nn.Conv2d(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                stride=stride, groups=conv_groups[3])

        self.conv_1=self.conv_1.to("cuda:0")
        self.conv_2 = self.conv_2.to("cuda:0")
        self.conv_3 = self.conv_3.to("cuda:0")
        self.conv_4 = self.conv_4.to("cuda:0")
        self.dan = DANetHead(planes // 4, planes // 4,norm_layer=nn.BatchNorm2d)

        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # 多尺度分组
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)  # [105, 64, 41, 41]
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])  # [105, 4, 16, 41, 41]



        # 使用DANet模块并保留所有注意力输出
        x1_dan, x1_sa, x1_sc = self.dan(x1)
        x2_dan, x2_sa, x2_sc = self.dan(x2)
        x3_dan, x3_sa, x3_sc = self.dan(x3)
        x4_dan, x4_sa, x4_sc = self.dan(x4)

        # 拼接DANet模块的输出特征
        x1_fusion = torch.cat((x1_dan, x1_sa, x1_sc), dim=1)  # [batch_size, 48, 1, 1]
        x2_fusion = torch.cat((x2_dan, x2_sa, x2_sc), dim=1)
        x3_fusion = torch.cat((x3_dan, x3_sa, x3_sc), dim=1)
        x4_fusion = torch.cat((x4_dan, x4_sa, x4_sc), dim=1)

        # 用卷积层进行融合
        conv_fusion = nn.Conv2d(576, self.split_channel, kernel_size=1)
        conv_fusion=conv_fusion.to("cuda:0")# [batch_size, 16, 1, 1]
        x1_fusion = conv_fusion(x1_fusion.float())
        x2_fusion = conv_fusion(x2_fusion.float())
        x3_fusion = conv_fusion(x3_fusion.float())
        x4_fusion = conv_fusion(x4_fusion.float())

        x_se = torch.cat((x1_fusion, x2_fusion, x3_fusion, x4_fusion), dim=1)
        # [256, 768, 1, 7]
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 7)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), dim=1)
        return out

