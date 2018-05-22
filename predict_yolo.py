import time
import cv2
import numpy as np
from chainer import serializers, Variable, cuda
import chainer.functions as F
import argparse
from model import YOLOv2, YOLOv2Predictor
from lib.utils import nms
from lib.utils import Box

fcn = False
class Predictor:
    def __init__(self, gpu=0):
        # hyper parameters
        weight_file = "./weight/fcn-un4-100"
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
        pred = self.model.predict(x)
        x, y, w, h, conf, prob = pred

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
        return nms_results

if __name__ == "__main__":
    # argument parse
    parser = argparse.ArgumentParser(description="predict image")
    parser.add_argument('--path', help="input image path")
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    image_file = args.path

    # read image
    orig_img = cv2.imread(image_file)

    predictor = Predictor(gpu=args.gpu)
    nms_results = predictor(orig_img)

    # draw result
    print(len(nms_results))
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle(
            orig_img,
            result["box"].int_left_top(), result["box"].int_right_bottom(),
            (255, 0, 255),
            3
        )
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)
    cv2.imwrite('out.png', orig_img)

