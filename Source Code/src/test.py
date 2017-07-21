# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 23:24:15 2017

@author: ZJ
"""
import random
from random import randint

import numpy as np
import math
import matplotlib.pyplot as plt
"""
设计了一个sin函数的数据集
training_data是训练集
test_data是测试集
"""
import random
import numpy as np
N=4000
def f(x):
	return float(math.sin(x))
	
def load_data():
    training_inputs = [np.array(random.uniform(-10, 10)) for each in range(1,N)]
#    training_results =training_inputs
    training_results = [f(training_inputs[y]) for y in range(0,N-1)]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.array(random.uniform(-10, 10)) for each in range(1,N)]
    test_results = [f(training_inputs[y]) for y in range(0,N-1)]
#    test_results=test_inputs
    test_data = zip(test_inputs, test_results)
    return (training_data,  test_data)





"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
一种用于实现前馈神经网络的随机梯度下降学习算法的模块。 
使用反向传播计算梯度。
"""


class Network(object):

    def __init__(self, sizes):
        """
        sizes是一个列表list,其中储存的是各层中神经元的数量
        网络的偏移和权重按均值为0方差为1的高斯分布初始化
        第一层被假设为输入层，没有设置偏差
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        """
        np.random.randn(y, x)生成y*x维的标准正态分布数组
        python切片语法为左开右闭，且从0开始
        """
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
        """
        np.dot表示点积(哈密顿乘积?),在此处可看做矩阵w与向量a转置的乘积
        输入的a是一个行向量，每次循环得到这一层神经元的输出值向量
        最后将得到一个具体的数值
        """

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """        
        使用迷你批次随机梯度下降训练神经网络。
        “training_data”是表示训练输入和所需输出的元组“(x,y)”的列表。
        epochs:总迭代次数
        mini_batch_size:采样时小批量数据的大小
        eta:学习速率
        test_data:测试集，评估神经网络的准确率
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, int(self.evaluate(test_data)), n_test)
            else:
                print "Epoch {0} complete".format(j)
        """
        先用random.shuffle()将training_data数据集打乱
        再在mini_batches中储存若干个大小为mini_batch_size的小批量数据
        """
        
    def update_mini_batch(self, mini_batch, eta):
        """
        利用BP算法对单个的小批量数据做权值和偏置的更改
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        """ 
        print("***************")
        print(self.weights)
        print(nabla_w)
        print("***************")
        """
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        """  
        print("***************")
        print(self.weights)
        print("***************")
        """
        """
        先初始化一个和self.biases格式相同的nabla_b
        nabla_w表示将每一次迭代的得到C对w的偏导叠加
        """
        
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        """
        先前向传播算出C的值
        """
        activation = x
        activations = [x] #储存各层神经元的激活值
        zs = [] # 储存各层神经元的Z值，Z=f(wx+b)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]  
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1])

        return (nabla_b, nabla_w)
          

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        """
        输出的a向量中值最大的下标，即为输出的识别出的数字
        """
  
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return sum(abs(x-y)<0.1 for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))




if __name__=='__main__':  
    training_data, test_data=load_data()  
    net=Network([1,6,1])

    net.SGD(training_data, 30, 1, 1, test_data=test_data)
   # print(net.biases)
    #print (net.weights)
    t_1 =[float(training_data[y][0]) for y in range(0, 300)]
    t_2 =[float(training_data[y][1]) for y in range(0, 300)]
    plt.scatter(t_1, t_2, marker='o', color='b')
    t = [float(test_data[y][0]) for y in range(0,300)]
    zt = [f(t[y]) for y in range(0, 300)]
   # plt.plot(t,zt, 'b*')
    yt =[float(net.feedforward(t[y]))for y in range(0, 300)]

    plt.scatter(t,zt,marker = 'x', color = 'm')
    plt.scatter(t,yt, marker='o', color='r')
  #plt.scatter(training_data[:1],training_data[:0])
  #  print(float(net.feedforward(2)))
    plt.show()
  #  print(t)
   # print(yt)