TASK: Perform K-NN Classification on the Iris dataset (K is an odd integer)

THIS IS AN EXAMPLE OF A LAZY LEARNER CLASSIFIER WHERE WE DO NOT BUILD A CLASSIFIER MODEL FIRST. RATHER WHAT EVER DATA IS AVAILABLE IS USED AS AND WHEN AN UNTESTED TUPLE BECOMES AVAILABLE.

Step 1: Normalize the data using Z-norm.

Step 2: Partition the data and the class labels into two sets (called the TRAIN SET and the TEST SET). The train set should contain approximately 75% of the tuples in the dataset and the corresponding labels. The remaining 25% should be part of the test set. 

STEP 3: For each tuple in the test set predict the class label according to the following algorithm.
  a) Find the distance to all points in the TRAIN SET form the current tuple.
  b) Find the K nearest train tuples.
  c) Find the majority class label among the K nearest tuple.
  e) Assign the majority label as the prediction for the current test tuple.
  f) perform the same for all test tuples.

STEP 4: From your prediction array, the array containg all your predictions, find the accuracy of the system. You can do this by finding the percentage of tuples where your prediction matched the actual class label.

STEP 5: Vary the value of K and plot a graph for the accuracy of the system for different values of K.

STEP 6: ***TO BE DONE AS AN ASSIGNMENT IN SPARE TIME***
Use PCA to reduce the number of features to "M"  and repeat the above experiment. Plot a curve of the accuracy in this case by varying K.

HINTS: numpy.argsort may be necessary.
