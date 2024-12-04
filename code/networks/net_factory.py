from networks.VNet import VNET
from networks.VNet import VNet, RECVNet
from networks.unet import UNet, UNet_2d


def net_factory(net_type="VNet", in_chns=1, class_num=2, mode="train", tsne=0):

    if net_type == "VNet" and mode == "FFC" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
        return net
    if net_type == "VNet" and mode == "REC" and tsne == 0:
        net = RECVNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
        return net
    if net_type == "VNet" and mode == "VNet" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
        return net

def BCP_net(in_chns=1, class_num=2, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net