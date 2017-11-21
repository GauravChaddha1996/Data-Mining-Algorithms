import numpy as ny
import numpy.random
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

iris = load_iris()
data = iris["data"]
labels = iris["target"]

# dicretisize target labels ,  target class : 1
labels = [0 if target == 0 else 1 for target in  iris.target]
        
normalised_data = MinMaxScaler().fit_transform(data) # normalise the data

train_X, test_X, train_y, test_y = train_test_split(normalised_data, labels, test_size = 0.25, random_state=33)


class NeuralNet(object):
    def __init__(self, hidden_nodes, learning_rate=0.01, output_nodes=1, epochs=1000):
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.activation_function = self.sigmoid
        self.epochs = epochs
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + ny.exp(-x))
    
    def init_weights(self):
        self.weights_input_to_hidden = ny.random.rand(self.n_attributes + 1, self.hidden_nodes)
        self.weights_hidden_to_output = ny.random.rand(self.hidden_nodes + 1, self.output_nodes)
    
    def forward_pass(self, row):
        hidden_inputs = ny.dot(row, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        hidden_outputs = ny.append(hidden_outputs, [1])
        
        final_inputs = ny.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = self.activation_function(final_inputs)
        
        return hidden_outputs, final_outputs
    
    def backward_pass(self, row, hidden_outputs, final_outputs, target):
        final_error = final_outputs*(1 - final_outputs)*(target - final_outputs)
        hidden_error = hidden_outputs*(1 - hidden_outputs)*final_error*self.weights_hidden_to_output
        hidden_error = hidden_error[:-1]
        
        self.weights_hidden_to_output += self.learning_rate*ny.dot(hidden_outputs, final_error)
        self.weights_input_to_hidden += self.learning_rate*ny.dot(row.T, hidden_error.T)
        
    
    def train(self, inputs, targets):
        self.n_attributes = inputs.shape[1]
        self.init_weights()
        for i in range(0, self.epochs):
            for row, target in zip(inputs, targets):
                row = ny.append(row, [1])
                row = row.reshape((1, self.n_attributes + 1))

                hidden_outputs, final_outputs = self.forward_pass(row)

                hidden_outputs = hidden_outputs.reshape((self.hidden_nodes + 1, 1))
                final_outputs = final_outputs.reshape((self.output_nodes, 1))

                self.backward_pass(row, hidden_outputs, final_outputs, target)
    
    def predict(self, inputs):
        results = []
        for row in inputs:
            row = ny.append(row, [1])
            row = row.reshape((1, self.n_attributes + 1))
            hidden_outputs, final_outputs = self.forward_pass(row)
            results.append(round(final_outputs[0]))
        return results
        
NN = NeuralNet(hidden_nodes=3)

NN.train(train_X, train_y)
predictions = NN.predict(test_X)

accuracy_score(predictions, test_y)*100

