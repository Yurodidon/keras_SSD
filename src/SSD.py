from keras.layers import *
from keras.models import Model
import keras.backend as K
from src.PriorBox import PriorBox
from src.Normalization import Normalization


def SSD(input_shape, num_classes=21):
    input_layer = Input(shape=input_shape)

    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_1')(input_layer)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1', padding='same')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_3')(conv5_2)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), name='pool5', padding='same')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), activation='relu', dilation_rate=(6, 6), padding='same', name='fc6')(pool5)
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    # Block 8
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)

    # Block 9
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7_1z)

    # Block 10
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv8_2')(conv8_1)


    pool6 = GlobalAveragePooling2D(name='global_pool')(conv8_2)

    img_size = (input_shape[1], input_shape[0])

    def predict_block(input, name, num_priors, min_size, aspect_ratio, max_size=None,
                    flip=True, clip=True, variances=[0.1, 0.1, 0.2, 0.2], do_norm=True):
        if(do_norm):
            input = Normalization(20, name=f'{name}_norm')(input)
        origin_loc = Conv2D(num_priors * 4, (3, 3), name=f"{name}_mbox_loc", padding='same')(input)
        origin_conf = Conv2D(num_priors * num_classes, (3, 3), name=f"{name}_mbox_conf", padding='same')(input)
        loc = Flatten(name=f"{name}_mbox_loc_flat")(origin_loc)
        conf = Flatten(name=f"{name}_mbox_conf_flat")(origin_conf)
        priorbox = PriorBox(img_size, min_size=min_size, aspect_ratio=aspect_ratio, max_size=max_size,
                                 flip=flip, clip=clip, variances=variances)(input)
        return conf, loc, priorbox

    # preidctions
    conv4_3_conf, conv4_3_loc, conv4_3_priorbox = predict_block(conv4_3, 'conv4_3', 3, 30.0, [2], do_norm=True)
    fc7_conf, fc7_loc, fc7_priorbox = predict_block(fc7, 'fc7', 6, 60.0, [2, 3], 114.0)
    conv6_2_conf, conv6_2_loc, conv6_2_priorbox = predict_block(conv6_2, 'conv6_2', 6, 114.0, [2, 3], 168.0)
    conv7_2_conf, conv7_2_loc, conv7_2_priorbox = predict_block(conv7_2, 'conv7_2', 6, 168.0, [2, 3], 222.0)
    conv8_2_conf, conv8_2_loc, conv8_2_priorbox = predict_block(conv8_2, 'conv8_2', 6, 222.0, [2, 3], 276.0)

    pool6_conf = Dense(6 * num_classes, name='pool6_conf')(pool6)
    pool6_loc = Dense(6 * 4, name='pool6_loc')(pool6)
    pool6_reshape = Reshape((1, 1, 256), name='pool6_reshape')(pool6)
    pool6_priorbox = PriorBox(img_size, min_size=276.0, max_size=330.0, aspect_ratio=[2, 3],
                              variances=[0.1, 0.1, 0.2, 0.2], name='pool6_priorbox')(pool6_reshape)

    conf = concatenate([conv4_3_conf,
                        fc7_conf,
                        conv6_2_conf,
                        conv7_2_conf,
                        conv8_2_conf,
                        pool6_conf],
                       axis=1, name='conf')
    loc = concatenate([conv4_3_loc,
                       fc7_loc,
                       conv6_2_loc,
                       conv7_2_loc,
                       conv8_2_loc,
                       pool6_loc],
                      axis=1, name='loc')
    priorbox = concatenate([conv4_3_priorbox,
                            fc7_priorbox,
                            conv6_2_priorbox,
                            conv7_2_priorbox,
                            conv8_2_priorbox,
                            pool6_priorbox],
                           axis=1, name='priorbox')

    num_boxes = K.int_shape(loc)[-1] // 4
    conf = Reshape((num_boxes, num_classes), name='conf_reshape')(conf)
    conf = Activation(activation='softmax', name='conf_softmax')(conf)
    loc = Reshape((num_boxes, 4), name='loc_reshape')(loc)
    prediction = concatenate([loc,
                              conf,
                              priorbox],
                              name='prediction', axis=2)

    model = Model(inputs=input_layer, outputs=prediction)

    return model

if __name__ == "__main__":
    model = SSD((300, 300, 3))
    model.summary()