import numpy as np
import chainer
from chainer.training import extensions
from model import YOLOv2, YOLOv2Predictor
from image_dataset import DatasetYOLO

n_classes_fcn = 7
n_classes_yolo = 2
n_boxes = 5
gpu = 0
epoch = 2
batchsize = 3
out_path = 'result'
initial_weight_file = None
weight_decay = 1e-5
test = False

yolov2 = YOLOv2(n_classes_fcn=n_classes_fcn, n_classes_yolo=n_classes_yolo, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2, FCN=False)
if initial_weight_file:
    serializers.load_hdf5(initial_weight_file, model)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.use_cleargrads()
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

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
trainer.extend(extensions.snapshot_object(model, 'model_snapshot_yolo_{.updater.epoch}'), trigger=(10, 'epoch'))
trainer.extend(extensions.ProgressBar(update_interval=10))

trainer.run()

