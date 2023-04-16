import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load your data into a pandas DataFrame
df = pd.read_csv('dbfinal.csv')

# group the data by sensitivity and breach year
grouped = df.groupby(['IsSensitive_True', 'Breach_year']
                     ).size().reset_index(name='count')

# pivot the data to create a matrix with breach year as the rows, sensitivity as the columns, and count as the values
pivoted = grouped.pivot(index='Breach_year',
                        columns='IsSensitive_True', values='count')

# create a heatmap for the data
sns.heatmap(pivoted, cmap='Blues', annot=True, fmt='g')

# set the axis labels and title
plt.xlabel('Sensitivity')
plt.ylabel('Breach Year')
plt.title('Sensitive vs Not Sensitive Data')

# show the plot
plt.show()