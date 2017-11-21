Logistic Regression Classifier
Perform Logistic Regression on the IRIS dataset in a One vs All (OvA) manner:

Aim: Write a classifier based on the principles of Regression Analysis that identifies a binary classification pattern. The classifier takes the form of a hyper plane in an attribute dimensional space and has the form w0+w1x1+w2x2+...+wmxm i.e. ∑mj=0wjxj
WARNING: Watch out for matrix dimensions when multiplying.

Method: 

1. Obtain a normalized copy of the iris dataset. The dataset contains two components X the attribute matrix and Y the label vector.
2. Ensure the target class (say "setosa") is converted to a one while all other classes are converted to a zero. We now have a "Binary Classification Problem."
3. Conventionally in ML problems we treat all vectors as column vectors. So each tuple is to be in the form of a column. This is easily obtained by transposing the matrix X.
4. Augment the data matrix by adding a cosmetic attribute which contains the value 1 for all tuples. This is the bias attribute to compensate for w0.
5. Optionally decompose your data into a train and test set.
6. Obtain an initial weight matrix containing "m+1" weights i.e. W=[w0,w1,w2,...wm]T. (Why m+1 when there are only m attributes?)
7. While not converged:
  a. Make your predictions as H=sigmoid(WTX), where the multiplication is a matrix mul, and sigmoid is defined as g(x)=\frac{1}{1+e^{-x}). (Mark the fact that the matrix multiplication generates the eq. of a plane in exactly the form that is required.)
  b. Calculate the penalty for the current set of weights as J(W)=−1N∑Ni=1[y(i)log(h(i))+(1−y(i))log(1−h(i))] (Can you justify the choice of the penalty function?)
  c. Calculate the error in each iteration and store in a buffer.
  d. Update the weights using the gradient descent strategy as wnewj=woldj−α∇J(W) where ∇J(W)=−(y(i)−h(i))x(i)j. Nabla is nothing but the derivative of the error function J. (BE VERY CAREFUL WITH THE SIGNS.) Alpha is a very small constant known as the learning rate, usually held at 0.01.
8. Make your final predictions with the converged set of weights as follows: h(i)=1 if sigmoid(WTX)≥0.5 else 0.
9. What is the accuracy of your classifier? How does it vary when you change the number of generations. Also plot error against generation and validate the results.
10. Use only two attributes to make a prediction and plot a prediction line showing the separation of the flowers in a scatter plot.
EXTENSIONS
How can we write the program so that it can calculate three set of weights for the three distinct classes at the same time. Then your prediction for an unknown flower would be max[p(class1),p(class2),p(class3)].

Often it is advisable to regularise weights so as to prevent overfitting. In this case the error function is obtained as J(W)=J(W)+1M∑Mj=0w2j. How would you update the weights if regulariaztion is to be applied to all the weights except the bias i.e. wnew0 is updated using the old formula but for all other weights a new update formula is required.
