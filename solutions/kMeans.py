import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn import datasets
%matplotlib inline 
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
data = iris.data
target = iris.target

class kMeanClusterer(object):
	def __init__(self,k,gens):
	    self.k = k
	    self.m = 0
	    self.n = 0 
	    self.gens = gens
	def load_data(self, data):
	    self.data = data
	    self.n = self.data.shape[0]
	    self.m = self.data.shape[1]
	    self.assignedCluster = np.zeros(self.n)
	def distance(self,r1,r2):
	    dist = 0.0
	    for i in range(len(r1)):
	        dist = dist + (r1[i]-r2[i])**2
	    return dist
	def findCluster(self,row):
	    tempDist = self.distance(self.mean[0],row)
	    minDistCluster = 0
	    for j in range(1,self.k):
	        curDist = self.distance(self.mean[j],row)
	        if curDist < tempDist:
	            minDistCluster = j
	            tempDist = curDist
	    return minDistCluster
	def cluster(self):
	    init_mean_rows = np.random.choice(self.n,self.k,replace = False)
	    self.mean = self.data[init_mean_rows]
	    tmpMean = np.zeros((self.k,self.m))
	    for g in range(self.gens):
	        # iterate over all rows and assign them a cluster
	        for i in range(self.n):
	            self.assignedCluster[i] = self.findCluster(self.data[i])
	        for j in range(self.k):
	            locs = np.where(self.assignedCluster == j)[0]
	            tmpMean[j,:] = np.mean(self.data[locs,:],axis=0)
	        self.mean = tmpMean
	    return self.assignedCluster
  
clusterer = kMeanClusterer(3,5)
clusterer.load_data(pp.StandardScaler().fit_transform(data[:,:2]))
predictions = clusterer.cluster()
err=0
for i in range(len(predictions)):
    err = err + (math.fabs(target[i]-predictions[i]))
err = (err/len(predictions))
acc = (1-err)*100
print("Accuracy is ",acc)

plt.scatter(clusterer.data[:,0],clusterer.data[:,1],c=clusterer.assignedCluster)


