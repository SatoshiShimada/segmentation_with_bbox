import numpy as np
import chainer
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from custom_yolov2 import *
from lib.image_generator import *

# hyper parameters
gpu = 0
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
momentum = 0.9
weight_decay = 0.005
learning_rate = 1e-3

#YOLO
yolo_max_batches = 3
n_classes_yolo = 2
batch_size_yolo = 5
n_boxes = 5
learning_schedules_yolo = {
    "0" : 1e-5,
    "500" : 1e-4,
    "10000" : 1e-5,
    "20000" : 1e-6
}

# FCN
fcn_max_batches = 3
n_classes_fcn = 7
batch_size_fcn = 5
learning_schedules_fcn = {
    "0" : 1e-3,
    "10000" : 1e-4
}

# load image generator
print("loading image generator...")
data_fcn = DataSetFCN()
data_yolo = DataSetYOLO()

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes_fcn=n_classes_fcn, n_classes_yolo=n_classes_yolo, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2)
#serializers.load_hdf5(initial_weight_file, model)

if gpu >= 0:
    cuda.get_device(gpu).use()
    model.to_gpu()

#optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer = optimizers.Adam(alpha=learning_rate)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train YOLO
for batch in range(yolo_max_batches):
    if str(batch) in learning_schedules_yolo:
        optimizer.alpha = learning_schedules_yolo[str(batch)]
    model.cleargrads()
    x, t = data_yolo.get_sample(batch_size_yolo)
    x = Variable(x)
    if gpu >= 0:
        x.to_gpu()

    loss = model(x, t, train=True, FCN=False)
    print("batch: %d lr: %f loss: %f" % (batch, optimizer.alpha, loss.data))
    loss.backward()
    optimizer.update()

    if (batch + 1) % 1000 == 0:
        model_file = "{}/yolo-{}.model".format(backup_path, batch + 1)
        print("saving model to {}".format(model_file))
        serializers.save_hdf5(model_file, model)

# start to train FCN
for batch in range(fcn_max_batches):
    if str(batch) in learning_schedules_fcn:
        optimizer.alpha = learning_schedules_fcn[str(batch)]
    model.cleargrads()
    x, t = data_fcn.get_sample(batch_size_fcn)
    x = Variable(x)
    t = Variable(t)
    if gpu >= 0:
        x.to_gpu()
        t.to_gpu()
    loss = model(x, t, train=True, FCN=True)
    print("batch: %d lr: %f loss: %f" % (batch, optimizer.alpha, loss.data))
    loss.backward()
    optimizer.update()

    if (batch + 1) % 1000 == 0:
        model_file = "{}/fcn-{}.model".format(backup_path, batch + 1)
        print("saving model to {}".format(model_file))
        serializers.save_hdf5(model_file, model)

print("saving model to {}/yolov2_final.model".format(backup_path))
serializers.save_hdf5("{}/yolov2_final.model".format(backup_path), model)

