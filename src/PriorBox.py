import keras.backend as K
import tensorflow as tf
import numpy as np
import keras

class PriorBox(keras.layers.Layer):
    def __init__(self, image_size, min_size, aspect_ratio, max_size=None,
                  flip=True, clip=True, variances=[0.1], **kwargs):
        self.image_size = image_size
        self.min_size = min_size
        self.max_size = max_size
        self.clip = clip
        self.flip = flip
        if(len(variances) == 1):
            self.variances = variances * 4
        elif(len(variances) == 4):
            self.variances = variances
        else:
            self.variances = [0.1, 0.1, 0.2, 0.2]

        self.wi, self.hi = 2, 1
        self.aspect_ratio = [1.0]
        if max_size:
            self.aspect_ratio.append(1.0)
        if aspect_ratio:
            for ar in aspect_ratio:
                if ar in self.aspect_ratio:
                    continue
                self.aspect_ratio.append(ar)
                if flip:
                    self.aspect_ratio.append(1.0 / ar)

        box_width, box_height = [], []
        for ar in self.aspect_ratio:
            if ar == 1 and len(box_width) == 0:
                box_width.append(self.min_size)
                box_height.append(self.min_size)
            elif ar == 1 and len(box_width) > 0:
                box_width.append(np.sqrt(self.min_size * self.max_size))
                box_height.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_width.append(self.min_size * np.sqrt(ar))
                box_height.append(self.min_size / np.sqrt(ar))
        self.box_width = 0.5 * np.array(box_width)
        self.box_height = 0.5 * np.array(box_height)

        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_p = len(self.aspect_ratio)
        width = input_shape[self.wi]
        height = input_shape[self.hi]
        num_boxes = num_p * width * height
        return input_shape[0], num_boxes, 8

    def call(self, input, **kwargs):

        # input_shape = K.int_shape(input)
        input_shape = input._keras_shape
        layer_width, layer_height = input_shape[self.wi], input_shape[self.hi]
        img_width, img_height = self.image_size[0], self.image_size[1]

        box_width = img_width / layer_width
        box_height = img_height / layer_height
        cbw, cbh = box_width / 2, box_height / 2
        linx = np.linspace(cbw, img_width - cbw, layer_width)
        liny = np.linspace(cbh, img_height - cbh, layer_height)
        origin = np.zeros(shape=(layer_width * layer_height * len(self.aspect_ratio), 8))

        p = 0
        for xi in range(linx.shape[0]):
            for yi in range(liny.shape[0]):
                for i in range(len(self.aspect_ratio)):
                    center_x, center_y = linx[xi], liny[yi]
                    bw, bh = self.box_width[i], self.box_height[i]
                    xmin, ymin = center_x - bw, center_y - bh
                    xmax, ymax = center_x + bw, center_y + bh
                    xmin, xmax = xmin / img_width, xmax / img_width
                    ymin, ymax = ymin / img_height, ymax / img_height
                    origin[p] = xmin, ymin, xmax, ymax, \
                                self.variances[0], self.variances[1], self.variances[2], self.variances[3]
                    p += 1

        if self.clip:
            origin = np.minimum(np.maximum(origin, 0.0), 1.0)
        origin = K.expand_dims(K.variable(origin), 0)
        pattern = [tf.shape(input)[0], 1, 1]
        tensor = tf.tile(origin, pattern)
        return tensor

