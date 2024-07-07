import numpy as np

def softmax(x):
    sum=0
    for i in range(len(x[0])):
        sum+=np.exp(x[0][i])
    for i in range(len(x[0])):
        x[0][i]=np.exp(x[0][i])/sum
    return x


def relu(x):
    for i in range(len(x[0])):
        if x[0][i]<=0:
            x[0][i]=0
    return softmax(x)
        

def forward_prop(W1,W2,a0,b1,b2):
    a1=relu(np.matmul(W1,a0)+b1)
    a2=relu(np.matmul(W2,a1)+b2)
    return a2

def compute_cost():
    pass


def back_prop():
    pass


W1=np.zeros((10,784))
W2=np.zeros((10,10))
x_train=np.array([[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   
  255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   
    0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   
    0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   
    0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   
  255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]],[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]])

y_train=np.array([0])
print(x_train.shape)