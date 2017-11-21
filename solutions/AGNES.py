import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline

data = StandardScaler().fit_transform(load_iris().data)
class D(object):
    def dis(self,r1,r2):
        dist = 0
        for i in range(len(r1)):
            dist = dist + (r1[i]-r2[i])*(r1[i]-r2[i])
        return math.sqrt(dist)
        
clusters = data
d = D()
while(len(clusters)>3):
    minDist = 100
    x = 0
    y = 0
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i==j:
                continue
            dist = d.dis(clusters[i],clusters[j])
            if dist<minDist:
                minDist = dist
                x = i
                y = j
    a = []
    for i in range(data.shape[1]):
        a.append((data[x,i]+data[y,i])/2)
    clusters = np.insert(clusters,1,a,0)
    clusters = np.delete(clusters,x,axis=0)
    clusters = np.delete(clusters,y,axis=0)
print(clusters)   

plt.scatter(clusters[:,0],clusters[:,1],c=load_iris().target[:3])

plt.scatter(data[:,0],data[:,1],c=load_iris().target[:])
