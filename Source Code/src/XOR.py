import numpy as np

def f(s):
    if (s>=0):
        return 1
    else :
        return 0;

def xor(X):
    a=np.zeros(4)
    a[0]=X[0]
    a[1]=X[1]
    a[2]=f(a[0]+a[1]-1.5)
    a[3]=f(a[0]+a[1]-2*a[2]-0.5)
    print(a)

X = np.array([0,0])
xor(X)
X = np.array([0,1])
xor(X)
X = np.array([1,0])
xor(X)
X = np.array([1,1])
xor(X)
#print(X[1])