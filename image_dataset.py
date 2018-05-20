import numpy as np
from PIL import Image
from chainer import dataset
import chainer
import cupy

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
            data = line.replace('\n', '').split(' ')
            data = [ float(d) for d in data ]
            #label, x, y, w, h = data
            ret.append(data)
        count = len(ret)
        while count < 10:
            ret.append((10.0, 0.0, 0.0, 0.0, 0.0))
            count += 1
        return np.array(ret, dtype=np.float32)
    elif mode == "data":
        img = Image.open(path)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return x

class DatasetYOLO(dataset.DatasetMixin):
    def __init__(self):
        self.train_dataset = "/home/satoshi/2018_04_28/images/"
        self.target_dataset = "/home/satoshi/2018_04_28/labels/"
        image_list = "image_list_yolo"
        with open(image_list, 'r') as fp:
            names = fp.readlines()
        self.data = []
        for name in names:
            name = name.replace('\n', '')
            xpath = self.train_dataset + name + ".jpg"
            ypath = self.target_dataset + name + ".txt"
            self.data.append((xpath, ypath))

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        xpath, ypath = self.data[i]
        x = load_data(xpath, mode="data")
        y = load_data(ypath, mode="label_yolo")
        #ground_truth = []
        #for t in y:
        #    ground_truth.append({
        #        "x": t[1],
        #        "y": t[2],
        #        "w": t[3],
        #        "h": t[4],
        #        "label": t[0],
        #    })
        #y = np.array(ground_truth)
        return x, y

class DatasetFCN(dataset.DatasetMixin):
    def __init__(self):
        self.train_dataset = "/home/satoshi/chainer_fcn2/exact/images/"
        self.train_dataset = "/home/satoshi/fcn/segd/gain/exact/images/"
        self.target_dataset = "/home/satoshi/chainer_fcn2/exact/labels/"
        self.target_dataset = "/home/satoshi/fcn/segd/gain/exact/labels/"
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

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        xpath, ypath = self.data[i]
        x = load_data(xpath, mode="data")
        y = load_data(ypath, mode="label_fcn")
        return x, y

