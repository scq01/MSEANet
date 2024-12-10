import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from attention import PAM_Module, CAM_Module, ECA_Attention, CBAM
from MSFblock import MSFblock


class CBR_Module(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, act=True,
                 is_separable=False):
        super(CBR_Module, self).__init__()
        self.conv = AtrousSeparableConvolution(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                               padding=padding, dilation=dilation,
                                               bias=False) if is_separable \
            else nn.Conv2d(in_planes, out_planes,
                           kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv  输入通道 = 输出通道 = groups
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv 1x1的卷积 调整通道
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Dilated_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Dilated_conv, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # conv2d->CBR  channel_attention->CAM
        self.c1 = CBR_Module(in_c, out_c, kernel_size=1, dilation=1, padding=0)
        self.c2 = CBR_Module(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = CBR_Module(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = CBR_Module(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = CBR_Module(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.msf = MSFblock(out_c)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        feature = self.msf(x1, x2, x3, x4)
        return feature



# Cross-Context Fusion
class CCF(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(CCF, self).__init__()

        self.conv1 = CBR_Module(in_channels[0], out_channels, 1, padding=0)
        self.conv2 = CBR_Module(in_channels[1], out_channels, 1, padding=0)
        self.conv3 = CBR_Module(in_channels[2], out_channels, 1, padding=0)
        self.scale_factor = [0.5, 2.0]
        self.dialation = Dilated_conv(3 * out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(input2)
        x2 = F.interpolate(x2, scale_factor=self.scale_factor[0], mode='bilinear', align_corners=True)

        x3 = self.conv3(input3)
        x3 = F.interpolate(x3, scale_factor=self.scale_factor[1], mode='bilinear', align_corners=True)
        feat = torch.cat([x1, x2, x3], dim=1)
        feat = self.dialation(feat)
        return feat + x1


# efe
class edge_feature_extractor(nn.Module):
    def __init__(self, in_c=64):
        super().__init__()
        self.down1 = nn.Sequential(
            CBR_Module(in_c, 128, kernel_size=3, padding=1),
            # nn.MaxPool2d(2, stride=2)
            AtrousSeparableConvolution(128, 128, kernel_size=3, padding=1, stride=2),
        )

        self.down2 = nn.Sequential(
            CBR_Module(128, 256, kernel_size=3, padding=1),
            # nn.MaxPool2d(2, stride=2),  # MaxPooling
            AtrousSeparableConvolution(256, 256, kernel_size=3, padding=1, stride=2),
            CBR_Module(256, 128, kernel_size=3, padding=1)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 1*1 cbr lateral
        self.c_cat = CBR_Module(128, 128, kernel_size=1, padding=0)
        # cbr fusion
        self.up_conv1 = CBR_Module(256, 128, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 1*1 cbr lateral
        self.c_cat2 = CBR_Module(64, 128, kernel_size=1, padding=0)
        # cbr fusion
        self.up_conv2 = CBR_Module(256, 128, kernel_size=3, padding=1)

        # edge pred.
        self.pred_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )
        self.eca = ECA_Attention(5)

    def forward(self, x):
        # bottom-up
        x1 = self.down1(x)
        x2 = self.down2(x1)

        # top-down and fusion
        x_up = self.up1(x2)
        x1_cat = self.c_cat(x1)
        x_up = torch.cat([x1_cat, x_up], dim=1)
        x_up = self.eca(x_up)
        x_up = self.up_conv1(x_up)
        x_up = self.up2(x_up)
        x_cat = self.c_cat2(x)
        x_up = torch.cat([x_cat, x_up], dim=1)
        x_up = self.eca(x_up)
        x_last = self.up_conv2(x_up)
        x_last_e = self.eca(x_last)

        # edge pred.
        x_pred = self.pred_block(x_last)
        return x_pred, x_last_e

# selective edge aware
class SEA(nn.Module):
    def __init__(self, in_channels):
        super(SEA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + 128, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv_3x3 = CBR_Module(128, 128, 3, 1, 1)
        self.cbam = CBAM(in_channels)
        self.eca = ECA_Attention(5)

    # 边缘特征图 特征图 深度监督输出图
    def forward(self, edge_info, x, pred):
        residual = x
        B, C, H, W = x.size()

        # high-frequency feature
        edge_feature = F.adaptive_avg_pool2d(edge_info, (H, W))
        edge_feature = self.conv_3x3(edge_feature)

        # reverse attention
        pred = torch.sigmoid(pred)
        background_att = 1 - pred
        background_x = x * background_att

        fusion_feature = torch.cat([background_x, edge_feature], dim=1)
        fusion_feature = self.eca(fusion_feature)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        return out


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            CBR_Module(in_channels, in_channels // 4, 3, 1, 1),
            CBR_Module(in_channels // 4, out_channels, 3, 1, 1)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = CBR_Module(in_channels, in_channels // 4, kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.block1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.block2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.block3 = backbone.layer2
        self.block4 = backbone.layer3
        self.block5 = backbone.layer4
        out_channels = 128
        self.edge_pred = edge_feature_extractor(64)

        self.ccf1 = CCF([256, 64, 512], out_channels)
        self.ccf2 = CCF([512, 256, 1024], out_channels)
        self.ccf3 = CCF([1024, 512, 2048], out_channels)

        self.pam = PAM_Module(512)
        self.cam = CAM_Module(512)
        self.x5_conv = CBR_Module(2048, 512, 3, 1, 1)
        self.up5 = nn.Sequential(
            CBR_Module(512, 512, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.out5 = Out(512, 1)
        self.sea4 = SEA(128)

        self.up4 = Up(640, 256)
        self.out4 = Out(256, 1)
        self.sea3 = SEA(128)

        self.up3 = Up(384, 128)
        self.out3 = Out(128, 1)
        self.sea2 = SEA(128)

        self.up2 = Up(256, 128)
        self.out2 = Out(128, 1)
        self.sea1 = SEA(64)

        self.up1 = Up(192, 64)
        self.out1 = Out(64, 1)
        self.out_1x1 = CBR_Module(out_channels * 3, out_channels, 1, padding=0)

        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBR_Module(64, 64, 1, 1, 0),
            CBR_Module(64, 64, 3, 1, 1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.block1(x)  # 1, 64, 128, 128
        x2 = self.block2(x1)  # 1 256 64 64
        x3 = self.block3(x2)  # 1 512 32 32
        x4 = self.block4(x3)  # 1 1024 16 16
        x5 = self.block5(x4)  # 1 2042 8 8
        edge_feature, edge_info = self.edge_pred(x1)  # 1 1 256 256，1 1 128 128
        x_g_1 = self.ccf1(x2, x1, x3)  # 1 128 64
        x_g_2 = self.ccf2(x3, x2, x4)  # 1 128 32
        x_g_3 = self.ccf3(x4, x3, x5)  # 1 128 16

        x5 = self.x5_conv(x5)  # 512 8 8

        x5_p = self.pam(x5)
        x5_c = self.cam(x5)
        x5_feature = x5_p + x5_c

        d5 = self.up5(x5_feature)  # 512 16 16
        out5 = self.out5(d5)  # 1 16 16
        ead4 = self.sea4(edge_info, x_g_3, out5)  # 128 16 16

        d4 = self.up4(d5, ead4)  # 256 32 32
        out4 = self.out4(d4)  # 1 32 32
        ead3 = self.sea3(edge_info, x_g_2, out4)  # 128 32 32

        d3 = self.up3(d4, ead3)  # 128 64 64
        out3 = self.out3(d3)  # 1 64 64
        ead2 = self.sea2(edge_info, x_g_1, out3)  # 128 64 64

        d2 = self.up2(d3, ead2)  # 128 128 128
        out2 = self.out2(d2)  # 1 128 128
        ead1 = self.sea1(edge_info, x1, out2)  # 64 128 128

        d1 = self.up1(d2, ead1)
        out1 = self.out1(d1)

        return out1, out2, out3, out4, out5, edge_feature


if __name__ == '__main__':
    input_tensor = torch.randn([1, 3, 256, 256]).cuda()
    ras = Model().cuda()
    out = ras(input_tensor)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
    pass
