import keras
import keras.backend as K

class Normalization(keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale

        super(Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = K.variable(self.scale * input_shape[3])
        self.trainable_weights.append(self.gamma)

    def call(self, inputs, **kwargs):
        output = K.l2_normalize(inputs, axis=3) * self.gamma
        return output

