from sklearn.datasets import load_iris
import numpy as ny
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
% matplotlib inline

iris = load_iris()
data = iris["data"][:,:2]
labels = iris["target"]
#data

N, M = data.shape
print(N, M)

Y = labels.reshape((N, 1))
Y = ny.array(Y==2).astype(int)
Y.shape

X = MinMaxScaler().fit_transform(data)
X = X.T
bias = ny.ones((1, N))
XB = ny.vstack((bias, X))

class LogisitcRegression:
    def __init__(self, alpha, gens):
        self.alpha = alpha
        self.gens = gens
    def sigmoid(self, x):
        return 1.0/(1+ny.exp(-x))
    def prediction(self):
        return self.sigmoid(self.weights.T.dot(self.X).T)
    def calculatePenalty(self):
        self.H = self.prediction()
        component1 = ny.sum(self.y*ny.log(self.H))
        component2 = ny.sum((1-self.y)*ny.log(1-self.H))
        return -ny.sum(component1 + component2)/self.N
    def updateWeights(self):
        nabla = self.alpha*((self.y - self.H).T.dot(self.X.T).T)
        #print(nabla)
        self.weights += nabla
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.M, self.N = X.shape
        self.weights = ny.zeros((self.M, 1))
        #self.H = self.predict()
        #print(H)  
        #J = self.calculatePenalty()
        #print(J)
        #self.updateWeights()
        #print(self.weights)
        self.buffer = []
        for g in ny.arange(self.gens):
            self.buffer.append(self.calculatePenalty())
            #print(self.calculatePenalty())
            self.updateWeights()   
            #print(self.weights)
    def plotErrorCurve(self):
        plt.plot(ny.array(self.buffer))
    def predict(self, t):
        return self.weights.T.dot(t)>=0.5
    def getWeights(self):
        return self.weights

clf = LogisitcRegression(alpha=0.001, gens=5000)
clf.fit(XB, Y)
clf.plotErrorCurve()
predictions = []
for i in ny.arange(XB.shape[1]):
    predictions.append(clf.predict(XB[:,i]))
print(ny.array(predictions).flatten().astype(int))

from sklearn.metrics import accuracy_score
score = accuracy_score(Y, predictions)
print(score)
X = X.T

#print(X.shape)
ax1_min, ax1_max = X[:,0].min(), X[:,0].max()
ax = ny.linspace(ax1_min, ax1_max, 50)
print(ax)

markers = ['*','^']
colours = ['red', 'black']
#for i in ny.arange(len(markers)):
    #plt.scatter(X[Y[:,0]==i, 0], X[Y[:,0]==i,1], marker=markers[i])
W = clf.getWeights()
line_vals = ny.zeros((50, 1))
plt.plot(ax, line_vals,'k', linewidth=2)
plt.show()    

markers = ['*','^']
colours = ['red', 'black']
for i in ny.arange(len(markers)):
    plt.scatter(X[Y[:,0]==i, 0], X[Y[:,0]==i,1], marker=markers[i])
W = clf.getWeights()
line_vals = -(W[0,0]+W[1,0]*ax)/W[2,0]
plt.plot(ax, line_vals,'k', linewidth=2)
plt.show()    

