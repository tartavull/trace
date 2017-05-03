from .conv_net import ConvNet, ConvArchitecture
from .unet import UNet, UNetArchitecture, ResVNet

from .conv_net import N4, N4_AFF, N4_3D, N4_DEEPER, N4_WIDENED, VD2D, VD2D_AFF, BN_VD2D, BN_VD2D_RELU, VD2D_3D
from .unet import RES_VNET, UNET_3D, UNET_3D_4LAYERS


MODEL_DICT = {
    'conv': ConvNet,
    'unet': UNet,
    'res_vnet': ResVNet,
}

ARCH_DICT = {
    'n4': N4,
    'n4_3d': N4_3D,
    'n4_widened': N4_WIDENED,
    'n4_deeper': N4_DEEPER,
    'vd2d': VD2D,
    'bn_vd2d': BN_VD2D,
    'bn_vd2d_relu': BN_VD2D_RELU,
    'vd2d_aff': VD2D_AFF,
    'res_vnet': RES_VNET,
    'unet_3d': UNET_3D,
    'unet_3d_4layers': UNET_3D_4LAYERS,
    'vd2d_3d': VD2D_3D,
}
