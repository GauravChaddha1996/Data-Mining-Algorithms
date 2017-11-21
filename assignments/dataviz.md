USE THE HOUSING DATASET TO ANSWER THE FOLLOWING QUESTIONS:

You can read the data using numpy's genfromtxt function. However we need to be careful, set the dtype parameter as None(why?). Also ensure you read the data from the second row onwards as the first row contains the column headings. Observe that the final data is returned in the form of strings. Create a matrix containing the first nine columns of the data and convert its data type to float. Use the ny.array command for doing so (setting the dtype to float). Make a separate vector of the last attribute and leave it as a categorical(string) attribute. You can also generate a list containing the column names from the first row of the data.

If all this is too time consuming please study about the pandas module and see what it offers.

1) Write a function that standardizes the numeric data using the mean and standard deviation. (Use the time function to time how long it takes to standardize all the data)
2) Draw Boxplots for all the numberic attributes in the data.
3) Plot the histograms for all the numeric attributes in a grid of size N x N. (Hint: Matplotlib does have a subplot command exactly like MATLAB).
4) It is generally believed that the price of houses in the state of California is affected by the location and population of a block. Use a scatter plot to plot the longitude of the districts against their latitudes. Make the size of the marker proportional to the population of the district and the colour of the marker proportionate to the average value of houses in the district. What is our observation. 
(HINT: The scatter function takes a lot of parameters, c to set the colour of a marker, s to set its size, plt.set_cmap('colourmap name')  sets a colourmap of your choice, alpha to set the transparency of points etc.)

5) It is also known that house prices are affected by the average income of a district and the number of rooms in a house. Can you answer whether house prices are more correlated to total rooms in houses in a block (i.e. the data given) or the bedrooms per house (needs to be calculated.)
