import numpy as np


def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x>0,1,0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def forward_prop(W1, W2, a0, b1, b2):
    z1 = np.matmul(a0, W1.T) + b1
    a1 = relu(z1)
    z2 = np.matmul(a1, W2.T) + b2
    a2 = softmax(z2)  # Changed from relu to softmax
    
    return a0, z1, a1, z2, a2 

def compute_cost(x_train,y_train,W1,W2,b1,b2,m):
    y_pred=[]
    for i in range(m):
        _,_,_,_,a2=forward_prop(W1,W2,x_train[i],b1,b2)
        y_pred.append(a2)
    
    y_pred = np.array(y_pred)
    cost=np.sum((y_pred-y_train)**2)/m
    return cost 





        



def back_prop(x_train,y_train,W1,W2,b1,b2,m):
 
    dW1=np.zeros_like(W1)
    dW2=np.zeros_like(W2)
    db1=np.zeros_like(b1)
    db2=np.zeros_like(b2)
    
    for i in range(m):
        a0,z1,a1,z2,a2=forward_prop(W1,W2,x_train[i],b1,b2)
        delta2=a2-y_train[i]
        delta1=np.dot(delta2,W2)*drelu(z1)

        dW2+=np.dot(delta2.T,a1)
        db2+=delta2
        dW1+=np.dot(delta1.T,a0)
        db1+=delta1

    return dW1/m,dW2/m,db1/m,db2/m

def grad_desc(iter,W1,W2,b1,b2,x_train,y_train,m,lr):
    
    

    for i in range(iter):
        dW1,dW2,db1,db2=back_prop(x_train,y_train,W1,W2,b1,b2,m)
        W1-=lr*dW1
        W2-=lr*dW2
        b1-=lr*db1
        b2-=lr*db2
       
    print(compute_cost(x_train,y_train,W1,W2,b1,b2,m))
    return W1,W2,b1,b2

W1=np.random.randn(10,784)
W2=np.random.randn(10,10)
b1=np.zeros((1,10))
b2=np.zeros((1,10))

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
m=y_train.shape[0]
print("m: ",m)


x_train = x_train.reshape(m,1,784)
y_train = y_train.reshape(m,1,10)

print("x: ",x_train[8])

x_train=x_train/255

print(compute_cost(x_train,y_train,W1,W2,b1,b2,m))
lr=0.01
iter=200

W1,W2,b1,b2=grad_desc(iter,W1,W2,b1,b2,x_train,y_train,m,lr)

x_test=np.array([[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,
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
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]])
x_test=x_test/255

_,_,_,_,y_test=forward_prop(W1,W2,x_test[0],b1,b2)

print(np.round(y_test))