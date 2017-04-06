from conv_net import *
from unet import *
from unet_mod_update import *

MODEL_DICT = {
    'conv': ConvNet,
    'unet': UNet,
    'res_vnet': ResVNet,
    'unet_mod': UNet_Mod,
    'unet_mod_nomax': UNet_Mod_nomax
}

PARAMS_DICT = {
    'n4': N4,
    'n4_3d': N4_3D,
    'n4_widened': N4_WIDENED,
    'n4_deeper': N4_DEEPER,
    'vd2d': VD2D,
    'bn_vd2d': BN_VD2D,
    'bn_vd2d_relu': BN_VD2D_RELU,
    'vd2d_bound': VD2D_BOUNDARIES,
    'res_vnet': RES_VNET,
    'unet_3d': UNET_3D,
    'unet_mod': UNET_MOD,
    'unet_mod_nomax': UNET_MOD_NOMAX
    'unet_3d_4layers': UNET_3D_4LAYERS,
}
