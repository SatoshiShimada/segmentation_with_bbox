from chainer import cuda

class DelGradient(object):
    """
    Delete gradient for fine-tuning.

    Reference: https://qiita.com/ysasaki6023/items/3040fe3896fe1ed844c3
    """
    name = 'DelGradient'
    def __init__(self, del_target):
        self.del_target = del_target

    def __call__(self, opt):
        for name, param in opt.target.namedparams():
            for d in self.del_target:
                if d in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad = 0

