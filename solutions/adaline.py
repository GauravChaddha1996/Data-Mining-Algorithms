import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris().data
data = StandardScaler().fit_transform(data)
target = load_iris().target
target = [0 if t==0 else 1 for t in target]

trainX, testX, trainY, testY = train_test_split(data,target,test_size = 0.25, random_state=42)


class AdalineNN(object):
    def __init__(self,epochs,alpha):
        self.epochs = epochs
        self.alpha = alpha
    def initWeight(self,m):
        self.weight = np.random.random_sample(m+1) 
        self.weight[m] = 1
    def sigmoid(self,h):
        return (1/(1+math.exp(-h))) 
    def train(self,data,target):
        cols = data.shape[1]
        self.initWeight(cols)
        for i in range(self.epochs):
            for j in range(len(data)):
                h = 0
                for k in range(cols):
                    h = h+ data[j][k]*self.weight[k]
                g = self.sigmoid(h)
                err = (target[j]-g)
                for k in range(cols):
                    self.weight[k] = self.weight[k] + self.alpha*err*data[j][k]
    def predict(self,data):
        predictions = np.zeros(len(data))
        cols = data.shape[1]
        for j in range(len(data)):
            h = 0
            for k in range(cols):
                h = h + data[j][k]*self.weight[k]
            g = self.sigmoid(h)
            predictions[j] = g
        return predictions
        
        

adaline = AdalineNN(epochs=10,alpha=0.2)
adaline.train(trainX[:,:2],trainY)
predictions = adaline.predict(testX[:,:2])

err = 0
for i in range(len(testY)):
    err = err + (testY[i]-predictions[i])
err = err/len(data)
acc = (1-err)*100
print("Accuracy is ",acc)
