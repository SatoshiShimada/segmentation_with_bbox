# coding: utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
from chainer import reporter
import chainer.links as L
import chainer.functions as F
from lib.utils import multi_box_iou
from lib.utils import Box
from lib.utils import box_iou
from lib.reorg import reorg

class YOLOv2(Chain):
    def __init__(self, n_classes_fcn, n_classes_yolo, n_boxes):
        super(YOLOv2, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, stride=1, pad=1, nobias=True),
            bn1=L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias1=L.Bias(shape=(64,)),
            conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1, nobias=True),
            bn2=L.BatchNormalization(64, use_beta=False, eps=2e-5),
            bias2=L.Bias(shape=(64,)),

            conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1, nobias=True),
            bn3=L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias3=L.Bias(shape=(128,)),
            conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1, nobias=True),
            bn4=L.BatchNormalization(128, use_beta=False, eps=2e-5),
            bias4=L.Bias(shape=(128,)),

            conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1, nobias=True),
            bn5=L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias5=L.Bias(shape=(256,)),
            conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1, nobias=True),
            bn6=L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias6=L.Bias(shape=(256,)),

            conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1, nobias=True),
            bn7=L.BatchNormalization(256, use_beta=False, eps=2e-5),
            bias7=L.Bias(shape=(256,)),

            pool1=L.Convolution2D(None, n_classes_fcn, 1, stride=1, pad=0),

            upsample3=L.Deconvolution2D(None, n_classes_fcn, ksize=16, stride=8, pad=4),

            conv14=L.Convolution2D(None, 1024, 3, stride=1, pad=1, nobias=True),
            bn14=L.BatchNormalization(1024, use_beta=False, eps=2e-5),
            bias14=L.Bias(shape=(1024,)),

            conv15=L.Convolution2D(None, n_boxes * (5 + n_classes_yolo), ksize=1, stride=1, pad=0),
        )
        self.n_boxes = n_boxes
        self.n_classes_fcn = n_classes_fcn
        self.n_classes_yolo = n_classes_yolo
        self.finetune = False

    def __call__(self, x, train=False):
        chainer.using_config('train', train)
        h = F.relu(self.bias1(self.bn1(self.conv1(x), finetune=self.finetune)))
        h = F.relu(self.bias2(self.bn2(self.conv2(h), finetune=self.finetune)))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.relu(self.bias3(self.bn3(self.conv3(h), finetune=self.finetune)))
        h = F.relu(self.bias4(self.bn4(self.conv4(h), finetune=self.finetune)))
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        high_resolution_feature = reorg(h)

        h = F.relu(self.bias5(self.bn5(self.conv5(h), finetune=self.finetune)))
        h = F.relu(self.bias6(self.bn6(self.conv6(h), finetune=self.finetune)))
        p3 = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)

        h = F.relu(self.bias7(self.bn7(self.conv7(p3), finetune=self.finetune)))
        h = F.concat((high_resolution_feature, h), axis=1)
        h = F.relu(self.bias14(self.bn14(self.conv14(h), finetune=self.finetune)))
        o_yolo = self.conv15(h)

        u3 = self.pool1(p3)

        h = u3
        o_fcn = self.upsample3(h)
        #o_fcn.unchain_backward()

        return o_fcn, o_yolo

class YOLOv2Predictor(Chain):
    def __init__(self, predictor, FCN=False):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125], [2.59375, 2.78125], [1.9375, 3.25]]
        self.thresh = 0.6
        self.seen = 0
        self.unstable_seen = 5000
        self.FCN = FCN

    def __call__(self, input_x, t, train=True):
        output_fcn, output_yolo = self.predictor(input_x, train=train)
        if self.FCN:
            if train:
                loss_fcn = F.softmax_cross_entropy(output_fcn, t)
                reporter.report({'loss': loss_fcn}, self)
                return loss_fcn
            else:
                loss = F.softmax(output_fcn)
                return loss
        batch_size, _, grid_h, grid_w = output_yolo.shape
        self.seen += batch_size
        x, y, w, h, conf, prob = F.split_axis(F.reshape(output_yolo, (batch_size, self.predictor.n_boxes, self.predictor.n_classes_yolo+5, grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
        x = F.sigmoid(x)
        y = F.sigmoid(y)
        conf = F.sigmoid(conf)
        prob = F.transpose(prob, (0, 2, 1, 3, 4))
        prob = F.softmax(prob)

        tw = np.zeros(w.shape, dtype=np.float32) # wとhが0になるように学習(e^wとe^hは1に近づく -> 担当するbboxの倍率1)
        th = np.zeros(h.shape, dtype=np.float32)
        tx = np.tile(0.5, x.shape).astype(np.float32) # 活性化後のxとyが0.5になるように学習()
        ty = np.tile(0.5, y.shape).astype(np.float32)

        if self.seen < self.unstable_seen:
            box_learning_scale = np.tile(0.1, x.shape).astype(np.float32)
        else:
            box_learning_scale = np.tile(0, x.shape).astype(np.float32)

        tconf = np.zeros(conf.shape, dtype=np.float32) # confidenceのtruthは基本0、iouがthresh以上のものは学習しない、ただしobjectの存在するgridのbest_boxのみ真のIOUに近づかせる
        conf_learning_scale = np.tile(0.1, conf.shape).astype(np.float32)

        tprob = prob.data.copy() # best_anchor以外は学習させない(自身との二乗和誤差 = 0)
        
        # 全bboxとtruthのiouを計算(batch単位で計算する)
        x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape[1:]))
        y_shift = Variable(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape[1:]))
        w_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape[1:]))
        h_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape[1:]))
        x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu()
        best_ious = []
        for batch in range(batch_size):
            #n_truth_boxes = len(t[batch])
            n_truth_boxes = int(sum( x[0] != 10.0 for x in t[batch])) # ??
            box_x = (x[batch] + x_shift) / grid_w
            box_y = (y[batch] + y_shift) / grid_h
            box_w = F.exp(w[batch]) * w_anchor / grid_w
            box_h = F.exp(h[batch]) * h_anchor / grid_h

            ious = []
            for truth_index in range(n_truth_boxes):
                t = chainer.cuda.to_cpu(t) # ??
                truth_box_x = Variable(np.broadcast_to(np.array(t[batch][truth_index][1], dtype=np.float32), box_x.shape))
                truth_box_y = Variable(np.broadcast_to(np.array(t[batch][truth_index][2], dtype=np.float32), box_y.shape))
                truth_box_w = Variable(np.broadcast_to(np.array(t[batch][truth_index][3], dtype=np.float32), box_w.shape))
                truth_box_h = Variable(np.broadcast_to(np.array(t[batch][truth_index][4], dtype=np.float32), box_h.shape))
                truth_box_x.to_gpu(), truth_box_y.to_gpu(), truth_box_w.to_gpu(), truth_box_h.to_gpu()
                ious.append(multi_box_iou(Box(box_x, box_y, box_w, box_h), Box(truth_box_x, truth_box_y, truth_box_w, truth_box_h)).data.get())  
            if ious:
                ious = np.array(ious)
                best_ious.append(np.max(ious, axis=0))
            else:
                best_ious.append(0)
        best_ious = np.array(best_ious)

        # 一定以上のiouを持つanchorに対しては、confを0に下げないようにする(truthの周りのgridはconfをそのまま維持)。
        tconf[best_ious > self.thresh] = conf.data.get()[best_ious > self.thresh]
        conf_learning_scale[best_ious > self.thresh] = 0

        # objectの存在するanchor boxのみ、x、y、w、h、conf、probを個別修正
        abs_anchors = self.anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            for truth_box in t[batch]:
                if truth_box[0] == 10.0: # ??
                    continue
                truth_w = int(float(truth_box[1]) * grid_w)
                truth_h = int(float(truth_box[2]) * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(Box(0, 0, float(truth_box[3]), float(truth_box[4])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                # objectの存在するanchorについて、centerを0.5ではなく、真の座標に近づかせる。anchorのスケールを1ではなく真のスケールに近づかせる。学習スケールを1にする。
                box_learning_scale[batch, truth_n, :, truth_h, truth_w] = 1.0 
                tx[batch, truth_n, :, truth_h, truth_w] = float(truth_box[1]) * grid_w - truth_w 
                ty[batch, truth_n, :, truth_h, truth_w] = float(truth_box[2]) * grid_h - truth_h
                tw[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box[3]) / abs_anchors[truth_n][0])
                th[batch, truth_n, :, truth_h, truth_w] = np.log(float(truth_box[4]) / abs_anchors[truth_n][1])
                tprob[batch, :, truth_n, truth_h, truth_w] = 0
                tprob[batch, int(truth_box[0]), truth_n, truth_h, truth_w] = 1

                # IOUの観測
                full_truth_box = Box(float(truth_box[1]), float(truth_box[2]), float(truth_box[3]), float(truth_box[4]))
                predicted_box = Box(
                    (x[batch][truth_n][0][truth_h][truth_w].data.get() + truth_w) / grid_w, 
                    (y[batch][truth_n][0][truth_h][truth_w].data.get() + truth_h) / grid_h,
                    np.exp(w[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][0],
                    np.exp(h[batch][truth_n][0][truth_h][truth_w].data.get()) * abs_anchors[truth_n][1]
                )
                predicted_iou = box_iou(full_truth_box, predicted_box)
                tconf[batch, truth_n, :, truth_h, truth_w] = predicted_iou
                conf_learning_scale[batch, truth_n, :, truth_h, truth_w] = 10.0

            # debug prints
            maps = F.transpose(prob[batch], (2, 3, 1, 0)).data
        #print("seen = %d" % self.seen)

        # loss計算
        tx, ty, tw, th, tconf, tprob = Variable(tx), Variable(ty), Variable(tw), Variable(th), Variable(tconf), Variable(tprob)
        box_learning_scale, conf_learning_scale = Variable(box_learning_scale), Variable(conf_learning_scale)
        tx.to_gpu(), ty.to_gpu(), tw.to_gpu(), th.to_gpu(), tconf.to_gpu(), tprob.to_gpu()
        box_learning_scale.to_gpu()
        conf_learning_scale.to_gpu()

        x_loss = F.sum((tx - x) ** 2 * box_learning_scale) / 2
        y_loss = F.sum((ty - y) ** 2 * box_learning_scale) / 2
        w_loss = F.sum((tw - w) ** 2 * box_learning_scale) / 2
        h_loss = F.sum((th - h) ** 2 * box_learning_scale) / 2
        c_loss = F.sum((tconf - conf) ** 2 * conf_learning_scale) / 2
        p_loss = F.sum((tprob - prob) ** 2) / 2
        #print("x_loss: %f  y_loss: %f  w_loss: %f  h_loss: %f  c_loss: %f   p_loss: %f" % 
        #    (F.sum(x_loss).data, F.sum(y_loss).data, F.sum(w_loss).data, F.sum(h_loss).data, F.sum(c_loss).data, F.sum(p_loss).data)
        #)
        reporter.report({'x_loss': F.sum(x_loss).data}, self)
        reporter.report({'y_loss': F.sum(y_loss).data}, self)
        reporter.report({'w_loss': F.sum(w_loss).data}, self)
        reporter.report({'h_loss': F.sum(h_loss).data}, self)
        reporter.report({'c_loss': F.sum(c_loss).data}, self)
        reporter.report({'p_loss': F.sum(p_loss).data}, self)

        loss_yolo = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        reporter.report({'loss': loss_yolo}, self)
        return loss_yolo

    def init_anchor(self, anchors):
        self.anchors = anchors

    def parse_yolo_output(self, input_x, yolo_output):
            output = yolo_output
            batch_size, input_channel, input_h, input_w = input_x.shape
            batch_size, _, grid_h, grid_w = output.shape
            x, y, w, h, conf, prob = F.split_axis(F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes_yolo+5, grid_h, grid_w)), (1, 2, 3, 4, 5), axis=2)
            x = F.sigmoid(x) # xのactivation
            y = F.sigmoid(y) # yのactivation
            conf = F.sigmoid(conf) # confのactivation
            prob = F.transpose(prob, (0, 2, 1, 3, 4))
            prob = F.softmax(prob) # probablitiyのacitivation
            prob = F.transpose(prob, (0, 2, 1, 3, 4))

            # x, y, w, hを絶対座標へ変換
            x_shift = Variable(np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape))
            y_shift = Variable(np.broadcast_to(np.arange(grid_h, dtype=np.float32).reshape(grid_h, 1), y.shape))
            w_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 0], (self.predictor.n_boxes, 1, 1, 1)), w.shape))
            h_anchor = Variable(np.broadcast_to(np.reshape(np.array(self.anchors, dtype=np.float32)[:, 1], (self.predictor.n_boxes, 1, 1, 1)), h.shape))
            x_shift.to_gpu(), y_shift.to_gpu(), w_anchor.to_gpu(), h_anchor.to_gpu() # !!!need to comment out when using cpu!!!
            box_x = (x + x_shift) / grid_w
            box_y = (y + y_shift) / grid_h
            box_w = F.exp(w) * w_anchor / grid_w
            box_h = F.exp(h) * h_anchor / grid_h
            return box_x, box_y, box_w, box_h, conf, prob

    def predict(self, input_x, both=False):
        output_fcn, output_yolo = self.predictor(input_x, train=False)
        if both:
            pred_fcn = F.softmax(output_fcn)
            pred_yolo = self.parse_yolo_output(input_x, output_yolo)
            return pred_fcn, pred_yolo
        elif self.FCN:
            loss = F.softmax(output_fcn)
            return loss
        else:
            return self.parse_yolo_output(input_x, output_yolo)

