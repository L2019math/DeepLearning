# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

class acs:
    def __init__(self , name):
        self.name = name
        print_hi(name)
    def fun(self): # 参数 ...
        print("Tt's my fun." + self.name)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("hello world")
    # ~~~~~~~~~~~~~~~常规操作~~~~~~~~~~~~~~~
    # name = input()
    # print(name)
    # print_hi('PyCharm')
    # s = acs("asd")
    # s.fun()

    # f 字符串，适用于python3.6 以后
    # name = "lzc"
    # print(f"Hello,{name}")

    # ~~~~~~~~~~~~~~~Numpy操作~~~~~~~~~~~~~~~
    # x=np.array([[1,2],[3,4]])
    # print(x)
    # print("x.shape= ",end=" ")
    # print(x.shape)
    # x = x.flatten() # 转换为一维数组
    # print("x转换为一维数组: ", end=" ")
    # print(x)
    # print(x[np.array([0,2,3])]) # 获取索引为 0，2，3 的元素

    # ~~~~~~~~~~~~~~~pyplot操作~~~~~~~~~~~~~~~
    # sin x 图像
    # x=np.arange(0,6,0.1)
    # y=np.sin(x)
    # plt.plot(x,y,label ="sin x")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()  # 显示图像的label
    # plt.show()

    # ~~~~~~~~~~~~~~~imread~~~~~~~~~~~~~~~
    # img = imread('avatar.jpg')
    # plt.imshow(img)
    # plt.show()

    # ~~~~~~~~~~~~~~~sigmoid~~~~~~~~~~~~~~~
    # sigmoid
    # def sigmoid(x):
    #     return 1 / (1+np.exp(-x))
    #
    # t = np.array([1.0,2.0,3.0])
    # print(t+1.0)
    # print(1.0/t)
    # print(sigmoid(t))

    # ~~~~~~~~~~~~~~~dot 操作~~~~~~~~~~~~~~~
    # np.dot 就是矩阵乘法
    # 常用大写表示矩阵
    # X = np.array([1,2])
    # Y = np.array([[1,3,5],[2,4,6]])
    # M = np.dot(X,Y)
    # print(X)
    # print("~~~~~~~~")
    # print(Y)
    # print("~~~~~~~~")
    # print(M)


    # 三层网络之间的信号传递

    # 输入层（0层）到第一层：
    # def sigmoid(x):
    #  return 1 / (1+np.exp(-x))
    # X = np.array([1.0,0.5])
    # W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    # B1 = np.array([0.1,0.2,0.3])
    #
    # A1 = np.dot(X,W1) + B1
    # Z1 = sigmoid(A1)
    # print(Z1)
    # print("~~~~~~~")
    # # 第一层到第二层：
    # W2 = np.array([[0.1,0.4] , [0.2,0.5] , [0.3,0.6]])
    # B2 = np.array([0.1,0.2])
    #
    # A2 = np.dot(Z1,W2) + B2
    # Z2 = sigmoid(A2)
    # print(Z2)
    # print("~~~~~~~")
    # # 第二层到第三层：
    # def identity_function(x):
    #     return x
    #
    # W3 = np.array([[0.1,0.3] , [0.2,0.4]])
    # B3 = np.array([0.1,0.2])
    #
    # A3 = np.dot(Z2,W3) + B3
    # Y = identity_function(A3)
    # print(Y)
    # print("~~~~~~~")
    #
    # # 规范化信息传递代码
    # def init_network():
    #     network = {}
    #     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    #     network['b1'] = np.array([0.1, 0.2, 0.3])
    #     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    #     network['b2'] = np.array([0.1, 0.2])
    #     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    #     network['b3'] = np.array([0.1, 0.2])
    #     return network
    #
    #
    # def forward(network, x):
    #     W1, W2, W3 = network['W1'], network['W2'], network['W3']
    #
    #
    #     b1, b2, b3 = network['b1'], network['b2'], network['b3']
    #     a1 = np.dot(x, W1) + b1
    #     z1 = sigmoid(a1)
    #     a2 = np.dot(z1, W2) + b2
    #     z2 = sigmoid(a2)
    #     a3 = np.dot(z2, W3) + b3
    #     y = identity_function(a3)
    #     return y
    #
    # network = init_network()
    # x = np.array([1.0, 0.5])
    # y = forward(network, x)
    # print(y)  # [ 0.31682708 0.69627909]

    # argmax:返回的是值最大的索引，axis=0：列，axis=1：行
    x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
    y = np.argmax(x,axis=1)

    print(y)