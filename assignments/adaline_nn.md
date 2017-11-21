LEARN AN ADALINE NN:

INPUT: Any two attributes of the iris dataset.
       The class labels with any one target class dicretized to 1 others as 0.

OUTPUT: The line of separation obtained finally.
        Final weights.
        Final accuracy

Method:

1. Decide on the number of epochs to use:
2. Start with a random weight vector (M+1, 1) (M is the number of attributes,one is for the bias);
3. For each epoch:
  For each training set data:
    a. Predict  h(i) = summation of w(i)x(i) for all inputs using bias.
    b. g(i) = sigmoid(h(i)).
    c. err = (t(i) - g(i)) (t(i) is the known class label for the ith tuple)
    d. w(j)(new) = w(j)(old) + learning_rate*err*x(i) (remember j is for weights while i is for tuples)
    e. Update the weights
4. Make the final predictions.
5. Report the accuracy.
6. Plot the separation line on the scatter plot.

EXTRA CREDIT:
Generalize the algorithm to learn the weights for all the classes simultaneoulsy.
