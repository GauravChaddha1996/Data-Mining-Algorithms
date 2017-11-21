HIerarchical Agglomerative Clustering (AGNES)
INPUTS: X - A Dataset of dimensions N x M
        k - The number of dimensions required

Method:
1. Start off by assuming every data point is a cluster of its own. (So you have N clusters initially).
2. Generate a distance matrix for all remaining clusters (This is time consuming)
3. While no_of_clusters_remaining > k:
  a. Find the two nearest remaining clusters.
  b. Merge them and update the center for the merged cluster. (Merging entails bringing the current clusters together in a new cluster, deleting one of the clusters and then updating the center of the new cluster as the mean of the two clusters you have just brought together. Hint LISTS in Python can be made up of other lists).
  c. Update the distance matrix. (You have one cluster less now)

OUTPUT: The final cluster centers
Observations: Do you think this is a slow approach. Why?

What about the validity of the results.
