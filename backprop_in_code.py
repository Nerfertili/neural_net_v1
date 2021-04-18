import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = 1 / (1 + np.exp(-X.dot(W3) - b3))
    #print(W2.shape)
    #A1 = Z1.dot(W2)+b2
    Z = 1 / (1 + np.exp(-Z1.dot(W2) - b2))
    A = Z.dot(W1) + b1
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1,keepdims = True)
    return Y,Z, Z1

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    
    return float(n_correct)/n_total

def cost(T,Y):
    tot = T* np.log(Y)
    return tot.sum()

def derivative_w3(X,Z,T,Y,W1,W2):
    #grad_of_w_given_layer = Z_previus.T.dot(delta_layer)
    #print('dim of w1',W1.shape,'dim of w2',W2.shape)
    dZ = (T - Y).dot(W1.T).dot(W2.T) * Z * (1 - Z)
    ret1 = X.T.dot((dZ))
    return ret1

def derivative_w2(X,Z,T,Y,W1):
    dZ = (T - Y).dot(W1.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)
    return ret2

def derivative_w1(Z,T,Y):
    return Z.T.dot(T-Y)

def derivative_b3(T, Y, W2, Z, W1):
    #grad_of_b_given_layer = sum(delta_layer, axis = 0)
    return ((T - Y).dot(W1.T).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def derivative_b2(T, Y, W1, Z):
    return ((T - Y).dot(W1.T) * Z * (1 - Z)).sum(axis=0)

def derivative_b1(T, Y):
    return (T - Y).sum(axis=0)

def main():

    #create the data
    Nclass = 500
    D = 2 # dimension of the input
    M = 3 # hidden layer size 
    K = 3 # number of classes

    X1 = np.random.randn(Nclass,D)+np.array([0,-3])
    X2 = np.random.randn(Nclass,D)+np.array([3,3])
    X3 = np.random.randn(Nclass,D)+np.array([-3,3])
    X = np.vstack([X1,X2,X3])

    Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
    N = len(Y)

    T = np.zeros((N,K))
    for i in range(N):
        T[i,Y[i]] = 1
    
    plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.2)
    plt.show()

    #random weights
    W3 = np.random.rand(D,M)
    b3 = np.random.randn(M)
    W2 = np.random.rand(3,M)
    b2 = np.random.randn(M)
    W1 = np.random.rand(M,K)
    b1 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []
    for epoch in range(100000):
        output, hidden,Z1 = forward(X,W1,b1,W2,b2,W3,b3)
        
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output,axis=1)
            r = classification_rate(Y,P)
            print('cost:',c,'classification rate', r)
            costs.append(c)

        W3 += learning_rate * derivative_w3(X,hidden,T,output,W1,W2)
        b3 += learning_rate * derivative_b3(T, output, W2, hidden, W1)
        W2 += learning_rate * derivative_w2(Z1,hidden,T,output,W1)
        b2 += learning_rate * derivative_b2(T, output, W1, hidden)
        W1 += learning_rate * derivative_w1(hidden,T,output)
        b1 += learning_rate * derivative_b1(T, output)

    plt.plot(costs)
    plt.show()
        



if __name__ == '__main__':
    main()