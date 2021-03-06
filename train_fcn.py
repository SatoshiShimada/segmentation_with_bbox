import numpy as np
import chainer
from chainer.training import extensions
from tiny_model import YOLOv2, YOLOv2Predictor
from image_dataset import DatasetFCN

n_classes_fcn = 7
n_classes_yolo = 2
n_boxes = 5
gpu = 1
epoch = 500
batchsize = 3
out_path = 'result/fcn-x4/'
initial_weight_file = 'result/yolo-x3/final.npz'
weight_decay = 1e-5
test = False
snapshot_interval = 10

yolov2 = YOLOv2(n_classes_fcn=n_classes_fcn, n_classes_yolo=n_classes_yolo, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2, FCN=True)
if initial_weight_file:
    chainer.serializers.load_npz(initial_weight_file, model)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.use_cleargrads()
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

train_data = DatasetFCN()
train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
if test:
    test_data = DatasetFCN()
    test_iter = chainer.iterators.SerialIterator(test_data, batchsize, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)

trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_path)
if test:
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']), trigger=(1, 'epoch'))
#trainer.extend(extensions.snapshot_object(model, 'model_snapshot_fcn_{.updater.epoch}'), trigger=(snapshot_interval, 'epoch'))
trainer.extend(extensions.ProgressBar(update_interval=10))

trainer.run()

chainer.serializers.save_npz(out_path + 'final.npz', model)

