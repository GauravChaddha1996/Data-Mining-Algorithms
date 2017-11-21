from sklearn.datasets import load_iris
import numpy as ny
iris = load_iris()
data = iris["data"]
labels = iris["target"]
print(iris.target_names)
print(iris.feature_names)
class_code = 1
refined_data = data[labels==1,2:]
print(refined_data[:5,:])
means = ny.mean(refined_data)
rep_means = ny.tile(means,(refined_data.shape[0], 1))
norm_data = refined_data - rep_means
sdevs = ny.std(refined_data, axis=0)
print(sdevs)
KP = ny.sum(norm_data[:,0]*norm_data[:,1])/(refined_data.shape[0]*ny.prod(sdevs))
print(KP)

