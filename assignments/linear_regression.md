Inputs : The file data0.txt containing data to perform linear regression

Output : A set of weights w, that enable the best linear prediction for the data.

STEPS:
1. Separate the data into two component X, and Y. Marks the fact that X is a biased set of attributes, with a bias attribute containing all ones.
2. Any line using the independent variable x is of the form w0 + w1x.
3. Therefore the prediction for the yi the values is obtained as w0 + w1xi.
4. Consequently the total error while predicting all the ys is given as J(w) = ∑(yi−xTiw)2
5. In terms of matrix multiplication this can also be written as (Y - Xw)T(Y-Wx).
6. Since the error is to be minimized, the derivative of this should equal zero.
7. The derivative of the expression in Step. 6 is given as XT(Y-Xw).
8. Find the set of w that produces the best linear prediction.
9. Plot the original values and the predicted values on the same plot.

HINTS: Vectors in numpy are not treated as one dimensional matrices by default. You will have to manually reshape them in to the desired shape. Use numpy.reshape for the same.
Numpy does matrix inverses using the submodule numpy.linalg
