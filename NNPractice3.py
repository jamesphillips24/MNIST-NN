import pandas as pd
import numpy as np
import random
import math

training_data = pd.read_csv("/Users/james/Visual-Studio/NN/mnist_train.csv")
training_data = np.array(training_data)

label = np.zeros((60000, 10))
for i in range(60000):
    label[i][training_data[i][0]] = 1
training_data = training_data[0:, 1:] / 255 # normalized

test_data = pd.read_csv("/Users/james/Visual-Studio/NN/mnist_test.csv")
test_data = np.array(test_data)

label1 = np.zeros((10000, 10))
for i in range(10000):
    label1[i][test_data[i][0]] = 1
test_data = test_data[0:, 1:] / 255 # normalized

def relu(z):
    return np.maximum(0.1*z, z)

def drelu(z):
    return (z > 0) * 1

def softmax(z):
    e_x = np.exp(z - np.max(z))
    return e_x/np.sum(e_x)

def ssr(z, labels):
    return np.sum(np.square(labels - z))

def crossEntropy(z, labels):
    i = np.where(labels == 1)[0]
    try:
        return -math.log(z[i] + 1e-8)
    except:
        print(z)
        print(labels)

def initialize(layer1_size, layer2_size, output_size):
    w1 = np.random.randn(layer1_size, np.shape(training_data[0])[0]) * np.sqrt(2/np.shape(training_data[0])[0]) # 8 x 784
    b1 = np.zeros(layer1_size)
    w2 = np.random.randn(layer2_size, layer1_size) * np.sqrt(2/np.shape(w1)[0]) # 8 x 8
    b2 = np.zeros(layer2_size)
    w3 = np.random.randn(output_size, layer2_size) * np.sqrt(2/np.shape(w2)[0]) # 10 x 8
    b3 = np.zeros(output_size)
    return w1, w2, w3, b1, b2, b3

def forward(dataset):
    z1 = np.dot(w1, dataset) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)

    return z1, z2, z3, a1, a2, a3

def backward(dataset, y, a1, a2, a3, z1, z2, z3, w1, w2, w3, dw1, db1, dw2, db2, dw3, db3):
    delta3 = a3 - y # Last layer node deltas
    dw3 += np.outer(delta3, a2)
    db3 += delta3

    delta2 = drelu(z2) * np.dot(w3.T, delta3.flatten())
    dw2 += np.outer(delta2, a1)
    db2 += delta2

    delta1 = drelu(z1) * np.dot(w2.T, delta2.flatten())
    dw1 += np.outer(delta1, dataset)
    db1 += delta1

    return dw1, db1, dw2, db2, dw3, db3



firstlayersize = 64
secondlayersize = 64
outputsize = 10
w1, w2, w3, b1, b2, b3 = initialize(firstlayersize, secondlayersize, outputsize)
batchsize = 100

for i in range(600):
    loss = 0
    dw1 = np.zeros((firstlayersize, 784))
    db1 = np.zeros(firstlayersize)
    dw2 = np.zeros((secondlayersize, firstlayersize))
    db2 = np.zeros(secondlayersize)
    dw3 = np.zeros((10, secondlayersize))
    db3 = np.zeros(outputsize)
    for j in range(batchsize):
        z1, z2, z3, a1, a2, a3 = forward(training_data[j + batchsize * i])
        loss += crossEntropy(a3, label[j + batchsize * i])
        dw1, db1, dw2, db2, dw3, db3 = backward(training_data[j + batchsize * i], label[j + batchsize * i], a1, a2, a3, z1, z2, z3, w1, w2, w3, dw1, db1, dw2, db2, dw3, db3)
    w1 -= 0.1 * dw1 / batchsize
    b1 -= 0.1 * db1 / batchsize
    w2 -= 0.1 * dw2 / batchsize
    b2 -= 0.1 * db2 / batchsize
    w3 -= 0.1 * dw3 / batchsize
    b3 -= 0.1 * db3 / batchsize
    print(loss)

print("Guessing!")
wrong = 0
for i in range(10000):
    z1, z2, z3, a1, a2, a3 = forward(test_data[i])
    guess = np.where(a3 == np.max(a3))
    if(label1[i][guess] == 0):
        print("Wrong!")
        wrong += 1
print(wrong)