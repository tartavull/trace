from conv_net import *

MODEL_DICT = {
    'conv': ConvNet,
}

PARAMS_DICT = {
    'n4': N4,
    'n4_widened': N4_WIDENED,
    'n4_deeper': N4_DEEPER,
    'vd2d': VD2D,
    'bn_vd2d': BN_VD2D,
    'bn_vd2d_relu': BN_VD2D_RELU,
    'vd2d_bound': VD2D_BOUNDARIES,
}
