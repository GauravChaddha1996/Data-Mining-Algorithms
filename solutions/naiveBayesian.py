from sklearn.datasets import load_iris
import numpy as ny
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
data = iris["data"]
labels = iris["target"]
Xnorm = StandardScaler().fit_transform(data)

trainX, testX, trainY, testY = train_test_split(Xnorm, labels, train_size=0.75, random_state=33)

class GaussianNBClassifier(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_classes = len(ny.unique(y))
        self.unique_classes = ny.unique(y)
        self.class_prob = []
        for i in ny.arange(self.num_classes):
            locs = ny.where(self.y == self.unique_classes[i])
            self.class_prob.append(len(locs[0]))
        self.class_prob = ny.array(self.class_prob)
    def get_uniue_classes(self):
        return self.unique_classes
    def get_class_probabilities(self):      
        return self.class_prob
    def get_num_classes(self):
        return self.num_classes
    def predict(self, t):
        tuple_class_prob = []
        tuple_total_prob = 0
        for c in ny.arange(self.num_classes): #starting off with one class at a time
            current_class = self.unique_classes[c]
            print(current_class)
            locs = ny.where(self.y == current_class)[0]
            print(locs)
            tmp = 1
            for a in ny.arange(len(t)):
                attr_mean = ny.mean(trainX[locs, a], axis=0)
                attr_sd = ny.std(trainX[locs, a], axis=0)
                print("Mean = ", attr_mean)
                print("SD = ", attr_sd)
                p = 1.0/(ny.sqrt(2*ny.pi*attr_sd))*ny.exp(-(t[a]-attr_mean)**2/(2*attr_sd**2))
                print("Attr Prob = ", p)
                tmp = tmp * p
            tuple_total_prob += tmp*self.class_prob[c]
            print("TUPLE PROB = ", tuple_total_prob)
            tuple_class_prob.append(tmp)
        tuple_class_prob /= tuple_total_prob
        print("TUPLE CLASS PROB = ", tuple_class_prob)
        s = ny.argsort(tuple_class_prob)
        print(self.unique_classes[s[-1]])
        return self.unique_classes[s[-1]]   
        
clf = GaussianNBClassifier(trainX, trainY)
print(clf.get_class_probabilities())
print(clf.get_num_classes())
print(clf.get_uniue_classes())

predictions = []
for i in ny.arange(testX.shape[0]):
    clf.predict(testX[i])
    predictions.append(clf.predict(testX[i]))
    #break

print(predictions)
score = accuracy_score(testY, predictions)
print(score)
