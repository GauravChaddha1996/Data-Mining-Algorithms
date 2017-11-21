Before starting off with the exercise please practice using numpy's where and delete functions. You should also explore the 3D plotting interface for matplotlib.

PERFORM K-Means Clustering (BASIC VERSION)

INPUT: 

The data set to be clustered (assumed to have "M" numeric attributes and "N" data points), which is assumed to be without a class label attribute. 
The number of clusters to create
STEPS:

1. Start by loading the iris dataset and normalizing it with the Standard Scaler.
2. Initialize "k" (where k is the number of clusters reqd.) centers each having "M" attributes.
3. Initialize an assignment array of size "N"
4. For each data point in the dataset:
  a. Compute the distances from all the centers.
  b. Sort the distances.
  c. Find the nearest cluster center.
  d. Update the assignment array with the nearest cluster center for the current point.
5. Recompute all the cluster centers by computing the mean of all the points which have been clustered with a particular cluster center.
6. Go back to step 4.
7. Continue till no changes are found in the assignment array.
8. Compute the correctness of your cluster using the homogeneity metrics found in scikit learn's metrics sub module.

WHAT ARE THE ISSUES THAT YOU FIND ? DID WE OBTAIN THE REQUISITE NUMBER OF CLUSTERS EVERY TIME?

SOLUTIONS ??
