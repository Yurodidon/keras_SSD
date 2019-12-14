import keras
import keras.backend as K
import numpy as np
from keras.engine.topology import InputSpec

class Normalization(keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale

        super(Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[3],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, 3)
        output *= self.gamma
        return output

