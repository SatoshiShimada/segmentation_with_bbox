import numpy as np
from PIL import Image
from lib.utils import *

def load_data(path, mode="label"):
    if mode == "label_fcn":
        img = Image.open(path)
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

class DataSetFCN(object):
    def __init__(self):
        self.image_size = (640, 480)
        self.train_dataset = "/home/satoshi/chainer_fcn2/exact/images/"
        self.target_dataset = "/home/satoshi/chainer_fcn2/exact/labels/"
        image_list = "image_list_fcn"
        with open(image_list, 'r') as fp:
            names = fp.readlines()
        self.data = []
        for name in names:
            name = name.replace('\n', '')
            xpath = self.train_dataset + name + ".png"
            ypath = self.target_dataset + name + ".png"
            self.data.append((xpath, ypath))
        self.num_data = len(self.data)

    def get_sample(self, batchsize):
        image_width, image_height = self.image_size
        xp = np
        x = xp.zeros((batchsize, 3, image_height, image_width), dtype=np.float32)
        y = xp.zeros((batchsize, image_height, image_width), dtype=np.int32)
        for b in range(batchsize):
            i = np.random.choice(range(self.num_data))
            xpath, ypath = self.data[i]
            x[b] = load_data(xpath, mode="data")
            y[b] = load_data(ypath, mode="label_fcn")
        x = np.array(x)
        y = np.array(y)
        return x, y

class DataSetYOLO(object):
    def __init__(self):
        self.image_size = (640, 480)
        self.train_dataset = "/home/satoshi/2018_04_28/images/"
        self.target_dataset = "/home/satoshi/2018_04_28/labels/"
        image_list = "image_list"
        with open(image_list, 'r') as fp:
            names = fp.readlines()
        self.data = []
        for name in names:
            name = name.replace('\n', '')
            xpath = self.train_dataset + name + ".jpg"
            ypath = self.target_dataset + name + ".txt"
            self.data.append((xpath, ypath))
        self.num_data = len(self.data)

    def get_sample(self, batchsize):
        image_width, image_height = self.image_size
        xp = np
        x = xp.zeros((batchsize, 3, image_height, image_width), dtype=np.float32)
        y = []
        for b in range(batchsize):
            i = np.random.choice(range(self.num_data))
            xpath, ypath = self.data[i]
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

