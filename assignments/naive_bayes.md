INPUT: The dataset consisting of data attributes and a class.

OUTPUT: A Classifier Model

STEPS:
1. Initially we assume that the dataset is comprised of only categorical attributes. There are "N" tuples, "C" classes and "M" attributes.
2. Our task is to find for an unknown tuple t, the probability P(Cj | t) for all Cj. We can then assign the class with the highest probability as the predicted class.
3. From Bayesian probability P(Cj | t) = P(t | Cj)*P(Cj) / P(t). (Mark the fact the denominator is a constant for all Cj).
4. In the naive case we assume the probability of a tuple is the product of the probabilities of the individual domain values for each of the attributes.
5. The result will often require the Laplace correction to be enforced for viability of the product operation.
6. In case the products become very small the multiplication can be replaced by taking the natural log of the products which transpires to the addition of the natural logs.
7. Find the accuracy of the model that you have built on the test set. (Expected accuracy ~75%)

How can we make the algorithm work for numeric attributes ? 

Hints: We can use the sklearn cross validation module to perform a train set / test set split.
