import torch
import torch.nn as nn


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv3d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm3d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, d, h, w = x.size()

        dims_to_fft = (-3, -2, -1)

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfftn(x, dim=dims_to_fft, norm="ortho")

        ffted = torch.cat(
            (ffted.real, ffted.imag), dim=1
        )

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        x, y = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(x, y)
        output = torch.fft.irfftn(ffted, dim=(2, 3, 4), norm="ortho", s=(d, h, w))

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv3d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        # 局部没分块卷积
        if self.enable_lfu:
            n, c, d, h, w = x.shape
            # split_no = 3
            # split_s_d = d // split_no
            # split_s_h = h // split_no
            # split_s_w = w // split_no
            # xs = torch.cat(torch.split(
            #     x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            # xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
            #                dim=1).contiguous()
            # xs = torch.cat(torch.split(xs, split_s_d, dim=-1),
            #                dim=1).contiguous()
            xs = self.lfu(x)
            # xs = xs.repeat(1, 1, 1, split_no, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        if in_cg == 0:
            in_cg = 1
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # print(in_cg, in_cl, out_cg, out_cl)

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l = nn.Conv3d(in_cl, out_cl, kernel_size,
                                 stride, padding, dilation, groups, bias)

        self.convl2g = nn.Conv3d(in_cl, out_cg, kernel_size,
                                 stride, padding, dilation, groups, bias)

        self.convg2l = nn.Conv3d(in_cg, out_cl, kernel_size,
                                 stride, padding, dilation, groups, bias)

        self.convg2g = SpectralTransform(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

        self.bn_l = nn.BatchNorm3d(int(out_channels * (1 - ratio_gout)))
        self.bn_g = nn.BatchNorm3d(int(out_channels * ratio_gout))

        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.size()[1] > 1:
            x_l, x_g = x.chunk(2, 1)
        else:
            x_l, x_g = x, x

        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl0 = self.convl2l(x_l)
            out_xl1 = self.convg2l(x_g)
            out_xl = out_xl0 + out_xl1
        if self.ratio_gout != 0:
            out_xg0 = self.convl2g(x_l)
            out_xg1 = self.convg2g(x_g)
            out_xg = out_xg0 + out_xg1

        out_xl = self.act_l(self.bn_l(out_xl))
        out_xg = self.act_g(self.bn_g(out_xg))
        output = torch.cat((out_xl, out_xg), dim=1)
        # print(output.size())
        return output


if __name__ == '__main__':
    device = torch.device("cpu")
    x = torch.randn((1, 16, 160, 192, 160))
    fourierUnit = FourierUnit(16, 16)
    y = fourierUnit(x)
    print(y.shape)
