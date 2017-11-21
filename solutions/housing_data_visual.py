import numpy as ny
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

data = pd.read_csv('housing.csv')
data.describe()

# renoving NA values
data = data.fillna(data.mean())
data.describe()


# Z scaling
z_scaler = lambda column: (column - column.mean())/ column.std()

numeric_columns = [column for column in data.columns][0:-1]
other_columns = [column for column in data.columns][-1]

scaled_data = pd.DataFrame()
scaled_data[numeric_columns] = data[numeric_columns].apply(z_scaler, axis=0)
scaled_data[other_columns] = data[other_columns]
scaled_data.describe()

figure = plt.figure(figsize=(16, 20))
ax1 = figure.add_subplot(211)
boxplot = ax1.boxplot([scaled_data.iloc[:, i].values for i in range(0, len(numeric_columns))],
                     labels=numeric_columns)
title = plt.title("Original Boxplot")

ax2 = figure.add_subplot(212)
boxplot2 = ax2.boxplot([scaled_data.iloc[:, i].values for i in range(0, len(numeric_columns))],
                     labels=numeric_columns)
title = plt.title("Scaled Boxplot")
ylim = plt.ylim([-3, 5])


fig = plt.figure(figsize=(15, 15))
for i in range(0,len(numeric_columns)):
    row_num = 330 + i + 1
    obj = fig.add_subplot(row_num)
    obj.hist(scaled_data.iloc[:, i].values)
    obj.set_title(numeric_columns[i])



plt.figure(figsize=(15, 10))

plot = plt.scatter(scaled_data["longitude"], 
                   scaled_data["latitude"], 
                   c=data["median_house_value"],
                   s=data["population"]/50,
                   alpha=0.1,
                   cmap=plt.get_cmap("jet"))
colorbar = plt.colorbar()
xlabel = plt.xlabel("Longitude")
ylabel = plt.ylabel("Latitude")
title = plt.title("Price vs Bedrooms vs household color:median price size:population")

plt.figure(figsize=(15, 10))
plot = plt.scatter(scaled_data["median_house_value"], 
                   scaled_data["median_income"])
xlabel = plt.xlabel("Median house value")
ylabel = plt.ylabel("Income")
title = plt.title("Price vs Bedrooms vs Income of population")
