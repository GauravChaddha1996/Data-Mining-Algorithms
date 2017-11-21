from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
data = iris["data"]
labels = iris["target"]
print(load_iris().target_names)
print(load_iris().feature_names)

for i in range(data.shape[1]):
    print(np.average(data[:,i]))
for i in range(data.shape[1]):
    print(np.var(data[:,i]))
    
plt.boxplot([data[:,i] for i in range(data.shape[1])])
plt.hist([data[:i] for i in range(0,2)])

print(data[:,1])
print(stats.zscore(data[:,1]))

plt.scatter(data[:,0],data[:,1],c=labels[:])
