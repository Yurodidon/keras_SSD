import src.SSD as SSD
from src.Utils import *
from src.Generators import *
from src.Loss import *
import pickle as pk
import keras

classes = 21
priors = pk.load(open('./priorboxes_300.ple', "rb"))
utils = BBoxUtility(classes, priors, 0.5, 0.45)

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.LearningRateScheduler(schedule),
              keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=3,
                                             save_weights_only=True)]

transponder = Yielder("../VOC2007/JPEGImages/", "../VOC2007/Annotations/",
                          (300, 300, 3), 32, utils, classes=VOC2007MAP, end=4800)
confrimer = Yielder("../VOC2007/JPEGImages/", "../VOC2007/Annotations/",
                          (300, 300, 3), 1, utils, classes=VOC2007MAP, start=5000)


model = SSD.SSD((300, 300, 3), classes)


from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

base_lr = 1e-4
optim = keras.optimizers.Adadelta(lr=base_lr)
model.compile(optimizer=optim, loss=MultiboxLoss(classes, neg_pos_ratio=2.0).compute_loss,
              metrics=['mae', 'categorical_accuracy', 'acc', r2_keras])


model.load_weights("weights_SSD300.hdf5", by_name=True, skip_mismatch=False)

freeze = ['block1_1', 'block1_2', 'pool1', 'block2_1',
          'block2_2', 'pool2',
          'block3_1', 'block3_2', 'block3_3', 'pool3']
for layers in model.layers:
    if layers.name in freeze:
        layers.trainable = False

epoch = 10

history = model.fit_generator(transponder.generate(), 4800 // 32, epoch, nb_val_samples=211,
                              nb_worker=32, use_multiprocessing=True,
                              validation_data=confrimer.generate(), verbose=1, callbacks=callbacks)

model.save_weights('./model_weights.wt')

