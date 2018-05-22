from chainer import serializers, Variable, cuda
import numpy as np
import cupy
from PIL import Image
import os
import argparse
import cv2

from model import YOLOv2, YOLOv2Predictor
from color_map import make_color_map

color_map = make_color_map()

fcn = True
class Predictor:
    def __init__(self, gpu=0):
        # hyper parameters
        weight_file = "./weight/model_snapshot_fcn_300"
        #weight_file = "./weight/model_snapshot_yolo_100"
        self.n_classes_fcn = 7
        self.n_classes_yolo = 2
        self.n_boxes = 5
        self.detection_thresh = 0.2
        self.iou_thresh = 0.1
        self.label_file = "./label.txt"
        with open(self.label_file, "r") as f:
            self.labels = f.read().strip().split("\n")

        # load model
        yolov2 = YOLOv2(n_classes_fcn=self.n_classes_fcn, n_classes_yolo=self.n_classes_yolo, n_boxes=self.n_boxes)
        model = YOLOv2Predictor(yolov2, FCN=fcn)
        serializers.load_npz(weight_file, model)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
        self.model = model
        self.gpu = gpu

    def __call__(self, img):
        orig_input_height, orig_input_width, _ = img.shape
        #img = cv2.resize(orig_img, (640, 640))
        input_height, input_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[np.newaxis, :, :, :]
        x = Variable(x_data)
        if self.gpu >= 0:
            x.to_gpu()
        pred = self.model.predict(x).data
        pred = pred[0].argmax(axis=0)
        if self.gpu >= 0:
            pred = cuda.to_cpu(pred)
        return pred

if __name__ == "__main__":
    # argument parse
    parser = argparse.ArgumentParser(description="predict image")
    parser.add_argument('--path', help="input image path")
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    image_file = args.path

    # read image
    orig_img = cv2.imread(image_file)

    predictor = Predictor(gpu=args.gpu)
    pred = predictor(orig_img)

    row, col = pred.shape
    dst = np.ones((row, col, 3))
    for i in range(args.classes):
        dst[pred == i] = color_map[i]
    img = Image.fromarray(np.uint8(dst))

    b,g,r = img.split()
    img = Image.merge("RGB", (r, g, b))

    trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
    w, h = img.size
    for x in range(w):
        for y in range(h):
            pixel = img.getpixel((x, y))
            if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0)or \
               (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                continue
            trans.putpixel((x, y), pixel)
    #o.paste(trans, (0,0), trans)

    if not os.path.exists("out"):
        os.mkdir("out")
    trans.save("out/pred.png")

    o = cv2.imread(image_file, 1)
    p = cv2.imread("out/pred.png", 1)

    pred = cv2.addWeighted(o, 0.4, p, 0.6, 0.0)
    cv2.imwrite("out/pred_{}.png".format("out"), pred)
    os.remove("out/pred.png")

