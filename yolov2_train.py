import numpy as np
import chainer
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from custom_yolov2 import *
from lib.utils import *
from lib.image_generator import *

# hyper parameters
gpu = 0
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 6
learning_rate = 1e-5
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 2
n_boxes = 5

# FCN
fcn_max_batches = 3000
n_classes_fcn = 7

#YOLO
n_classes_yolo = 2
yolo_max_batches = 30000

# load image generator
print("loading image generator...")
data_fcn = DataSetFCN()
data_yolo = DataSetYOLO()

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes_fcn=n_classes_fcn, n_classes_yolo=n_classes_yolo, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2)
#serializers.load_hdf5(initial_weight_file, model)

model.predictor.train = True
model.predictor.finetune = False
if gpu >= 0:
    cuda.get_device(gpu).use()
    model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train YOLO
for batch in range(yolo_max_batches):
    model.cleargrads()
    x, t = data_yolo.get_sample(batch_size)
    x = Variable(x)
    if gpu >= 0:
        x.to_gpu()

    loss = model(x, t, train=True, FCN=False)
    print("batch: %d lr: %f loss: %f" % (batch, optimizer.lr, loss.data))
    loss.backward()
    optimizer.update()

# start to train FCN
for batch in range(yolo_max_batches):
    model.cleargrads()
    x, t = data_fcn.get_sample(batch_size)
    x = Variable(x)
    t = Variable(t)
    if gpu >= 0:
        x.to_gpu()
        y.to_gpu()
    loss = model(x, t, train=True, FCN=True)
    print("batch: %d lr: %f loss: %f" % (batch, optimizer.lr, loss.data))
    loss.backward()
    optimizer.update()

print("saving model to %s/yolov2_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_final.model" % (backup_path), model)

if gpu >= 0:
    model.to_cpu()
serializers.save_hdf5("%s/yolov2_final_cpu.model" % (backup_path), model)

