import time
import os
import argparse
import cv2
from PIL import Image
import numpy as np
import chainer
from chainer import serializers, Variable, cuda
import chainer.functions as F

from tiny_model import YOLOv2, YOLOv2Predictor
from lib.utils import nms
from lib.utils import Box
from color_map import make_color_map

color_map = make_color_map()

chainer.using_config('train', False)
chainer.using_config('enable_backprop', False)

fcn = False
class Predictor:
    def __init__(self, gpu=0, weight=None):
        # hyper parameters
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
        if weight:
            serializers.load_npz(weight, model)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
        self.model = model
        self.gpu = gpu

    def __call__(self, img):
        orig_input_height, orig_input_width, _ = img.shape
        #img = cv2.resize(orig_img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[np.newaxis, :, :, :]
        x = Variable(x_data)
        if self.gpu >= 0:
            x.to_gpu()
        pred_fcn, pred_yolo = self.model.predict(x, both=True)

        pred_fcn = pred_fcn[0].data.argmax(axis=0)
        if self.gpu >= 0:
            pred_fcn = cuda.to_cpu(pred_fcn)
        x, y, w, h, conf, prob = pred_yolo

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes_yolo, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh
        if self.gpu >= 0:
            x = cuda.to_cpu(x)
            y = cuda.to_cpu(y)
            w = cuda.to_cpu(w)
            h = cuda.to_cpu(h)
            conf = cuda.to_cpu(conf)
            prob = cuda.to_cpu(prob)
            detected_indices = cuda.to_cpu(detected_indices)

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "label": self.labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i]*orig_input_width,
                            y[detected_indices][i]*orig_input_height,
                            w[detected_indices][i]*orig_input_width,
                            h[detected_indices][i]*orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = nms(results, self.iou_thresh)
        return pred_fcn, nms_results

if __name__ == "__main__":
    # argument parse
    parser = argparse.ArgumentParser(description="predict image")
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--path', help="input image path")
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--weight', default='weight.npz', type=str)
    args = parser.parse_args()
    image_file = args.path

    predictor = Predictor(gpu=args.gpu, weight=args.weight)
    times = []
    # read image
    files = os.listdir(args.path)
    for image_file in files:
        if not 'png' in image_file:
            continue
        print(image_file)
        ooo = cv2.imread(args.path + image_file)
        orig_img = cv2.resize(ooo, (320, 240))

        start_time = time.time()
        pred_fcn, nms_results = predictor(orig_img)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        # FCN
        row, col = pred_fcn.shape
        dst = np.ones((row, col, 3))
        for i in range(args.classes):
            dst[pred_fcn == i] = color_map[i]
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

        oo = cv2.imread(args.path + image_file, 1)
        pp = cv2.imread("out/pred.png", 1)
        o = cv2.resize(oo, (320, 240))
        p = cv2.resize(pp, (320, 240))
        os.remove("out/pred.png")

        fcn_img = cv2.addWeighted(o, 0.4, p, 0.6, 0.0)
        cv2.imwrite('out/' + image_file, fcn_img)

        # draw YOLO result
        print(len(nms_results))
        for result in nms_results:
            left, top = result["box"].int_left_top()
            cv2.rectangle(
                fcn_img,
                result["box"].int_left_top(), result["box"].int_right_bottom(),
                (255, 0, 255),
                3
            )
            text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
            cv2.putText(fcn_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            print(text)
        cv2.imwrite('out/' + image_file, fcn_img)

    print(times[0])
    times = times[1:]
    print(sum(times) / len(times))
    print(max(times))
    print(min(times))

