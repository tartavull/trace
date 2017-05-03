import tensorflow as tf

from trace.common import *
from trace.models import ConvNet, N4, VD2D

from .averager import ModelAverager

ENSEMBLE_METHOD_DICT = {
    ModelAverager.name: ModelAverager
}

N4_3_TEST = [
    ComponentParams(
        id='n4_1',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=500,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_2',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=500,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_3',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=500,
            output_size=101,
        )
    )
]

N4_3 = [
    ComponentParams(
        id='n4_1',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=30000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_2',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=30000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_3',
        model=ConvNet,
        architecture=N4,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=30000,
            output_size=101,
        )
    )
]

VD2D_5 = [
    ComponentParams(
        id='vd2d_1',
        model=ConvNet,
        architecture=VD2D,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_2',
        model=ConvNet,
        architecture=VD2D,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_3',
        model=ConvNet,
        architecture=VD2D,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_4',
        model=ConvNet,
        architecture=VD2D,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_5',
        model=ConvNet,
        architecture=VD2D,
        training_params=TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iterations=50000,
            output_size=101,
        )
    ),
]

ENSEMBLE_PARAMS_DICT = {
    'n4_3': N4_3,
    'n4_3_test': N4_3_TEST,
    'vd2d_5': VD2D_5,
}
