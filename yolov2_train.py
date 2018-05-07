import numpy as np
import chainer
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from custom_yolov2 import *
from lib.utils import *
from lib.image_generator import *

# hyper parameters
gpu = 0
image_list = "image_list"
train_dataset = "/home/satoshi/2018_04_28/images/"
target_dataset = "/home/satoshi/2018_04_28/labels/"
backup_path = "backup"
backup_file = "%s/backup.model" % (backup_path)
batch_size = 16
max_batches = 30000
learning_rate = 1e-5
learning_schedules = { 
    "0"    : 1e-5,
    "500"  : 1e-4,
    "10000": 1e-5,
    "20000": 1e-6 
}

lr_decay_power = 4
momentum = 0.9
weight_decay = 0.005
n_classes = 2
n_boxes = 5

# load image generator
print("loading image generator...")
generator = ImageGenerator(image_list, train_dataset, target_dataset)

# load model
print("loading initial model...")
yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
model = YOLOv2Predictor(yolov2)
#serializers.load_hdf5(initial_weight_file, model)

model.predictor.train = True
model.predictor.finetune = False
cuda.get_device(gpu).use()
model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
print("start training")
for batch in range(max_batches):
    if str(batch) in learning_schedules:
        optimizer.lr = learning_schedules[str(batch)]

    # generate sample
    x, t = generator.generate_samples(batch_size)
    x = Variable(x)
    x.to_gpu()

    # forward
    loss = model(x, t)
    print("batch: %d lr: %f loss: %f" % (batch, optimizer.lr, loss.data))

    # backward and optimize
    #optimizer.zero_grads()
    model.cleargrads()
    loss.backward()
    optimizer.update()

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

print("saving model to %s/yolov2_final.model" % (backup_path))
serializers.save_hdf5("%s/yolov2_final.model" % (backup_path), model)

model.to_cpu()
serializers.save_hdf5("%s/yolov2_final_cpu.model" % (backup_path), model)

