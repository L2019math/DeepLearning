

# SGD 随机梯度下降法
# W = W - n grad(L,W)
import numpy as np


class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# Momentum 动量
# v = av - n grad(L,W)
# W = W + v
# v 相当于物理上的速度
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# AdaGrad 为参数的每个元素适当地（adaptive）调整学习率
# h = h + dot( grad(L,W) ,grad(L,W) )
# W = W - n * 1 / sqrt(h) * grad(L,W)
# h 为过去所有梯度的平方和
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
