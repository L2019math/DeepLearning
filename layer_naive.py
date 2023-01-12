import numpy as np

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 乘法的 backward propagation 是交换 x,y
    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy

# 加法层
class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out

    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy

# Relu层
# x<= 0 : 0 , x>0 : x
class Relu:
    def __init__(self):
        self.mask = None

    # x 是 NumPy 数组
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        # 将x<=0的索引所在的元素置为零
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None
    # 1/(1+exp(-x))
    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    # dout * y(1-y)
    def back(self,dout):
        dx = dout *( 1.0 - self.out)*self.out
        return dx

# Affine层
# Y = X*W + B
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        return dx

def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x))
    return y

# one-hot 类型的交叉熵
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size())
        y = t.reshape(1,y.size())

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 标签形式的交叉熵
def cross_entropy_error_2(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size())
        y = t.reshape(1, y.size())

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arrange(batch_size)] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None # softmax 的输出
        self.t = None # one-hot vector

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx
    
#----------- example: buy_apple

# 超参数
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

print(price) # 220

# 此外，关于各个变量的导数可由backward()求出。
dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple,dapple_num,dtax) # 2.2,110,200

#----------- example: buy_apple_orange

# 超参数
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple,apple_num) # 1 step
orrange_price = mul_orange_layer.forward(orange,orange_num) # 2 step
all_price = add_apple_orange_layer.forward(apple_price,orrange_price) # 3 step
price = mul_tax_layer.forward(all_price,tax) # 4 step

# backward
dprice = 1
dall_price,dtax = mul_tax_layer.backward(dprice) # -1 step
dapple_price,dorrange_price = add_apple_orange_layer.backward(dall_price) # -2 step
dorrange,dorange_num = mul_orange_layer.backward(dorrange_price) # -3 step
dapple,dapple_num = mul_apple_layer.backward(dapple_price) # -4 step

print(price) # 715
print(dapple_num,dapple,dorrange,dorange_num,dtax) # 110 2.2 3.3 165 650