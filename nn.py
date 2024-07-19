
import numpy as np
import tensorflow as tf

def onehot(y):
    return np.eye(10)[y]

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return np.where(x>0,1,0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def forward_prop(W1, W2, a0, b1, b2):
    z1 = np.dot(a0, W1.T) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2.T) + b2
    a2 = softmax(z2) 
    
    return a0, z1, a1, z2, a2 

def compute_cost(x_train, y_train, W1, W2, b1, b2, m):
    _, _, _, _, a2 = forward_prop(W1, W2, x_train, b1, b2)
    cost = -np.sum(y_train * np.log(a2 + 1e-8)) / m
    return cost

def back_prop(x_train, y_train, W1, W2, b1, b2, m):
    a0, z1, a1, z2, a2 = forward_prop(W1, W2, x_train, b1, b2)
    
    delta2 = a2 - y_train
    dW2 = np.dot(delta2.T, a1) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m
    
    delta1 = np.dot(delta2, W2) * drelu(z1)
    dW1 = np.dot(delta1.T, a0) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m
    
    return dW1, dW2, db1, db2



def grad_desc(iter,W1,W2,b1,b2,x_train,y_train,m,lr):
    for i in range(iter):
        dW1,dW2,db1,db2=back_prop(x_train,y_train,W1,W2,b1,b2,m)
        W1-=lr*dW1
        W2-=lr*dW2
        b1-=lr*db1
        b2-=lr*db2
        if i%10==0:
            print("Iteration:",i)
            print(compute_cost(x_train,y_train,W1,W2,b1,b2,m))
       
    print(compute_cost(x_train,y_train,W1,W2,b1,b2,m))
    return W1,W2,b1,b2


W1=np.random.randn(10,784)*0.01
W2=np.random.randn(10,10)*0.01
b1=np.zeros((1,10))
b2=np.zeros((1,10))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.ceil(x_train / 255.0), np.ceil(x_test / 255.0)




reshaped_grid_str = np.array2string(x_train[4].reshape(1,784), separator=', ')


print("testing3: ",reshaped_grid_str)


x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

m=60000
iter=1000
lr=0.2

x_train = x_train[0:m]
y_train = y_train[0:m]

y_train=onehot(y_train)
y_test=onehot(y_test)

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

W1,W2,b1,b2=grad_desc(iter,W1,W2,b1,b2,x_train,y_train,m,lr)

np.savetxt('W1.txt', W1)
np.savetxt('W2.txt', W2)
np.savetxt('b1.txt', b1)
np.savetxt('b2.txt', b2)

acc = 0

for i in range(10000):
    _, _, _, _, y_pred = forward_prop(W1, W2, x_test[i], b1, b2)
    y_pred = np.argmax(y_pred)  
    y_true = np.argmax(y_test[i])  

    if y_pred == y_true:
        acc += 1

print(f"Accuracy: {acc/10000 * 100}%")


x_eg=np.array([[[0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
   38,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,  76, 255, 255, 255, 255,
  255, 255, 255, 255,  38,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 110,  76,
  255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 132,   0,   0,   0,
    0,   0,   0,  76, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255, 220,   0,   0,   0,   0,
    0,   0,   0,   0,  38, 136, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 228, 255, 164,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 152, 255, 114,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255,  44,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 142,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 158,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 218,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 228, 255, 114,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 255, 255, 224,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,  38, 255, 255,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 255, 255, 218,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255,  66,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0, 228, 255, 255,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,  76, 255, 255, 202,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 255, 255, 255,  66,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0, 228, 255, 255, 255, 255, 114,   0,
    0,   0,  76, 255, 255, 255, 220,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 220,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
  255, 255, 255, 154,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]])

x_eg=np.ceil(x_eg/255)
print("draw shape:",x_eg[0])

print(y_train[2])

_,_,_,_,y_eg=forward_prop(W1, W2, x_eg[0], b1, b2)

print(y_eg)


