import math


class CosineWarmupScheduler:

    def __init__(self, base_lr: float, init_lr: float, total_epoch: int, warmup_epoch: int):
        """

        Args:
            base_lr: the initial learning rate after warmup
            init_lr: the initial learning rate before warmup
            total_epoch: total number of epochs, including warmup epochs
            warmup_epoch: number of warmup epochs
        """
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.total_epoch = total_epoch
        self.warmup_epoch = warmup_epoch

    def __call__(self, epoch=None):

        if epoch <= self.warmup_epoch:
            lr = (self.base_lr - self.init_lr) / self.warmup_epoch * epoch + self.init_lr
        else:
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * (epoch - self.warmup_epoch) / (self.total_epoch - self.warmup_epoch))) + 1e-12
        lr /= self.init_lr  # since LambdaLR sets the current_learning rate to `curr_lr = curr_lr * lr` instead of using the number here.
        return lr
