import numpy as np
import pickle as ple

config300 = [
    (300, 300),
    {"name" : "conv4_3",  "size" : 38, "num_priors" : 3, "min_size" : 30.0,
     "max_size" : None,  "aspect_ratio" : [2]},
    {"name" : "fc7",      "size" : 19, "num_priors" : 6, "min_size" : 60.0,
     "max_size" : 114.0, "aspect_ratio" : [2, 3]},
    {"name" : "conv6_2",  "size" : 10, "num_priors" : 6, "min_size" : 114.0,
     "max_size" : 168.0, "aspect_ratio" : [2, 3]},
    {"name" : "conv7_2",  "size" : 6,  "num_priors" : 6, "min_size" : 168.0,
     "max_size" : 222.0, "aspect_ratio" : [2, 3]},
    {"name" : "conv8_2", "size" : 6,  "num_priors" : 6, "min_size" : 222.0,
     "max_size" : 276.0, "aspect_ratio" : [2, 3]},
    {"name" : "pool6",    "size" : 1,  "num_priors" : 6, "min_size" : 276.0,
     "max_size" : 330.0, "aspect_ratio" : [2, 3]}
]


def prior(config, img_size):
    # clip & filp always true
    size = config["size"]
    num_priors = config["num_priors"]
    min_size = config['min_size']
    max_size = config['max_size']
    ars = config['aspect_ratio']
    aspect_ratio = [1.0]
    if max_size:
        aspect_ratio.append(1.0)
    if ars:
        for ar in ars:
            if ar in aspect_ratio:
                continue
            aspect_ratio.append(ar)
            aspect_ratio.append(1.0 / ar)

    box_width, box_height = [], []
    for ar in aspect_ratio:
        if ar == 1 and len(box_width) == 0:
            box_width.append(min_size)
            box_height.append(min_size)
        elif ar == 1 and len(box_width) > 0:
            box_width.append(np.sqrt(min_size * max_size))
            box_height.append(np.sqrt(min_size * max_size))
        elif ar != 1:
            box_width.append(min_size * np.sqrt(ar))
            box_height.append(min_size / np.sqrt(ar))
    box_width = 0.5 * np.array(box_width)
    box_height = 0.5 * np.array(box_height)

    cell_width = img_size[0] / size
    cell_height = img_size[1] / size
    cbw, cbh = cell_width / 2, cell_height / 2
    linx = np.linspace(cbw, img_size[0] - cbw, size)
    liny = np.linspace(cbh, img_size[1] - cbh, size)
    origin = np.zeros(shape=(size * size * len(aspect_ratio), 8))

    p = 0
    for xi in range(linx.shape[0]):
        for yi in range(liny.shape[0]):
            for i in range(len(box_width)):
                center_x, center_y = linx[xi], liny[yi]
                bw, bh = box_width[i], box_height[i]
                xmin, ymin = center_x - bw, center_y - bh
                xmax, ymax = center_x + bw, center_y + bh
                xmin, xmax = xmin / img_size[0], xmax / img_size[0]
                ymin, ymax = ymin / img_size[1], ymax / img_size[1]
                origin[p] = xmin, ymin, xmax, ymax, 0.1, 0.1, 0.2, 0.2
                p += 1
    origin = np.minimum(np.maximum(origin, 0.0), 1.0)
    return origin

def create(config):
    img_size = config[0]
    priorbox, have = None, False
    for i in range(1, len(config)):
    # for i in [1]:
        ret = prior(config[i], img_size)
        if(not have):
            priorbox = ret
            have = True
        else:
            priorbox = np.vstack((priorbox, ret))
    return priorbox

if __name__ == "__main__":
    ret = create(config300)
    # for each in ret:
    #     for e in each:
    #         print(e, end=' ')
    #     print()
    f = open('priorboxes_300.ple', 'wb')
    ple.dump(ret, f)
    f.close()

    # ret = ple.load(open('prior_boxes_ssd300.pkl', 'rb'))
    # print(ret.shape)
    # for each in ret:
    #     for e in each:
    #         print(e, end=' ')
    #     print()
