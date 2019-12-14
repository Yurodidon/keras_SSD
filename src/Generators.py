import numpy as np
import os
import scipy.misc
import random
from xml.etree import ElementTree
import cv2 as cv
from keras.applications.imagenet_utils import preprocess_input


VOC2007MAP = {
                "aeroplane": 0,
                "bicycle": 1,
                "bird": 2,
                "boat": 3,
                "bottle": 4,
                "bus": 5,
                "car": 6,
                "cat": 7,
                "chair": 8,
                "cow": 9,
                "diningtable": 10,
                "dog": 11,
                "horse": 12,
                "motorbike": 13,
                "person": 14,
                "pottedplant": 15,
                "sheep": 16,
                "sofa": 17,
                "train": 18,
                "tvmonitor": 19
            }

RBCMAP = {'RBC' : 0}

def xmlExtractor(xml, convert):
    parse = ElementTree.parse(xml)
    root = parse.getroot()
    image_size = root.find('size')
    image_width = float(image_size.find('width').text)
    image_height = float(image_size.find('height').text)
    out_box, out_ann = [], []
    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        xmin = float(bndbox.find('xmin').text) / image_width
        ymin = float(bndbox.find('ymin').text) / image_height
        xmax = float(bndbox.find('xmax').text) / image_width
        ymax = float(bndbox.find('ymax').text) / image_height

        box_now = [xmin, ymin, xmax, ymax]
        name = object.find('name').text

        def one_hot(name):
            l = [0] * len(convert.items())
            l[convert[name]] = 1
            return l

        c = one_hot(name)
        out_box.append(box_now)
        out_ann.append(c)

    box = np.asarray(out_box)
    ann = np.asarray(out_ann)
    return np.hstack([box, ann])

# Generator without data augmentation
class Yielder(object):
    def __init__(self, ImagePath, AnnPath, image_size, batch_size, utils,
                 classes=VOC2007MAP, start=0, end=None, WE=False):
        self.ImagePath = ImagePath
        self.AnnPath = AnnPath
        self.image_size = image_size
        self.batch_size = batch_size
        self.utils = utils
        self.classes = classes
        self.start = start
        self.end = end
        self.images = os.listdir(self.ImagePath)
        self.images = [self.ImagePath + self.images[x] for x in range(len(self.images))]
        if(WE):
            raise KeyboardInterrupt

    def generate(self):
        if self.end == None:
          self.end = len(self.images)
        while True:
            out_images, out_ann = [], []
            random.shuffle(self.images)
            self.annotations = [self.AnnPath + x.split('/')[-1].split('.')[0] + ".xml" for x in self.images]
            cnt = 0
            for index in range(self.start, self.end):
                # img = scipy.misc.imread(self.images[index]).astype('float32')
                # img = scipy.misc.imresize(img, (self.image_size[0], self.image_size[1])).astype('float32')
                img = cv.imread(self.images[index]).astype('float32')
                img = cv.resize(img, (self.image_size[0], self.image_size[1])).astype('float32')
                ann = xmlExtractor(self.annotations[index], self.classes)
                ann = self.utils.assign_boxes(ann)
                out_images.append(img)
                out_ann.append(ann)
                if (len(out_images) == self.batch_size):
                    tmp, tmp_a = np.array(out_images).astype('float32'), np.array(out_ann).astype('float32')
                    out_images, out_ann, cnt = [], [], 0
                    yield tmp / 255., tmp_a
