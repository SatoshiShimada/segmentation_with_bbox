import numpy as np
import chainer
from chainer.training import extensions
from model import YOLOv2, YOLOv2Predictor
from image_dataset import DatasetYOLO
from lib import DelGradient

n_classes_fcn = 7
n_classes_yolo = 2
n_boxes = 5
gpu = 0
epoch = 500
batchsize = 3
out_path = 'result/yolo-un1'
initial_weight_file = 'result/fcn-un5/model_snapshot_fcn_100'
weight_decay = 1e-5
test = False
clear_gradient_layer = 250
snapshot_interval = 10

yolov2 = YOLOv2(n_classes_fcn=n_classes_fcn, n_classes_yolo=n_classes_yolo, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2, FCN=False)
if initial_weight_file:
    chainer.serializers.load_npz(initial_weight_file, model)

optimizer = chainer.optimizers.Adam()
#optimizer = chainer.optimizers.MomentumSGD(lr=1e-5)
optimizer.setup(model)
optimizer.use_cleargrads()
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
target = [ "conv{}".format(i+1) for i in range(clear_gradient_layer) ] + [ "bn{}".format(i+1) for i in range(clear_gradient_layer) ] + [ "bias{}".format(i+1) for i in range(clear_gradient_layer) ]
print(target)
optimizer.add_hook(DelGradient(target))

train_data = DatasetYOLO()
train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
if test:
    test_data = DatasetYOLO()
    test_iter = chainer.iterators.SerialIterator(test_data, batchsize, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)

trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_path)
if test:
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/x_loss', 'main/y_loss', 'main/w_loss', 'main/h_loss', 'main/c_loss', 'main/p_loss', 'elapsed_time']), trigger=(1, 'epoch'))
trainer.extend(extensions.snapshot_object(model, 'model_snapshot_yolo_{.updater.epoch}'), trigger=(snapshot_interval, 'epoch'))
trainer.extend(extensions.ProgressBar(update_interval=10))

trainer.run()

