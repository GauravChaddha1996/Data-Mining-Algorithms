import numpy as ny
from sklearn.datasets import load_iris
from scipy import stats
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline

iris_data_raw = load_iris().data
iris_data_classes = load_iris().target

# Normalize data using Z-Norm
data = stats.zscore(iris_data_raw)
target=iris_data_classes
trainX,testX,trainY,testY = train_test_split(data,target)

class KNN(object):
    def __init__(self,k):
        self.k = k
    def distance(self,row1,row2):
        dist = 0 
        for i in range(len(row1)):
            dist = dist + (row1[i]-row2[i])*(row1[i]-row2[i])
        return math.sqrt(dist)
    def getNeighbors(self,row,trainData,trainTarget):
        neighbors = []
        for i in range(len(trainData)):
            dist = self.distance(row,trainData[i])
            neighbors.append((trainData[i],dist,trainTarget[i]))
        neighbors.sort(key=lambda tup: tup[1])
        result = []
        for i in range(self.k):
            result.append(neighbors[i])
        return result
    def getResponse(self,neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][2]
            if response in classVotes:
                classVotes[response] +=1
            else:
                classVotes[response] =1
        classVotes2 = sorted(classVotes.items(),reverse=True)
        return classVotes2[0][0]
    def predict(self,trainData,trainTarget,testData):
        predictions = []
        for i in range(len(testData)):
            neighbors = self.getNeighbors(testData[i],trainData,trainTarget)
            result = self.getResponse(neighbors)
            predictions.append(result)
        return predictions

    
    
accc=[]
kk =[]
for k in range(3,22):
    knn = KNN(k)
    predictions = knn.predict(trainX,trainY,testX)
    err = 0 
    for i in range(len(predictions)):
        err = err + (predictions[i]-testY[i])
    err = (err/len(predictions))
    acc = (1-err)*100
    kk.append(k)
    accc.append(acc)
plt.scatter(kk,accc)
