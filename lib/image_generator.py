import numpy as np
from PIL import Image
from lib.utils import *

def load_data(path, mode="label"):
    if mode == "label_fcn":
        y = np.asarray(img, dtype=np.int32)
        y[y == 255] = -1
        return y
    elif mode == "label_yolo":
        with open(path, 'r') as fp:
            lines = fp.readlines()
        ret = []
        for line in lines:
            label, x, y, w, h = line.replace('\n', '').split(' ')
            ret.append((label, x, y, w, h))
        return ret
    elif mode == "data":
        img = Image.open(path)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return x

class ImageGenerator(object):
    def __init__(self, image_list, image_path, label_path, n_classes=2, image_size=(640, 480)):
        self.image_list = image_list
        self.image_path = image_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.image_size = image_size
        with open(self.image_list, 'r') as fp:
            names = fp.readlines()
        data = []
        for name in names:
            name = name.replace('\n', '')
            xpath = train_dataset + name + ".jpg"
            ypath = target_dataset + name + ".txt"
            data.append((xpath, ypath))
        self.num_data = len(data)

    def generate_samples(self, batchsize):
        image_width, image_height = self.image_size
        xp = np
        x = xp.zeros((batchsize, 3, image_height, image_width), dtype=np.float32)
        y = []
        #y = xp.zeros((batchsize, self.n_classes, image_height, image_width), dtype=np.float32)
        for b in range(batchsize):
            i = np.random.choice(range(len(names)))
            x[b] = load_data(xpath, mode="data")
            ts = load_data(ypath, mode="label_yolo")
            ground_truth = []
            for t in ts:
                ground_truth.append({
                    "x": t[1],
                    "y": t[2],
                    "w": t[3],
                    "h": t[4],
                    "label": t[0],
                })
            y.append(ground_truth)
        x = np.array(x)
        y = np.array(y)
        return x, y

