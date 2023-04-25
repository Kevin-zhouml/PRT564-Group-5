import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# read the merged CSV file into a pandas dataframe
df = pd.read_csv('databreaches650.csv')

# create a list of unique data classes
unique_values = set()
for index, row in df.iterrows():
    data_classes = row['DataClasses'].replace('[','').replace(']','').split(',')
    for value in data_classes:
        unique_values.add(value.strip().strip("'"))

# create a matrix where each row represents a breach and each column represents a data class
matrix = []
for index, row in df.iterrows():
    data_classes = row['DataClasses'].replace('[','').replace(']','').split(',')
    matrix_row = [0]*len(unique_values)
    for value in data_classes:
        matrix_row[list(unique_values).index(value.strip().strip("'"))] = 1
    matrix.append(matrix_row)

#print(matrix)

# cluster the data classes using k-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(matrix)

# add a new column to the dataframe with the cluster assignments
df['Cluster'] = kmeans.labels_

# create a new DataFrame that contains the count of breaches for each cluster
cluster_counts = df['Cluster'].value_counts().reset_index()

# rename columns
cluster_counts.columns = ['Cluster', 'Count']

# sort values by cluster number in ascending order
cluster_counts = cluster_counts.sort_values('Cluster')


# plot the distribution using a bar chart
plt.figure(figsize=(10,6))
plt.bar(cluster_counts['Cluster'], cluster_counts['Count'])
plt.xticks(range(n_clusters), fontsize=12)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Number of breaches', fontsize=14)
plt.title('Distribution of breaches based on DataClasses clusters', fontsize=16)
plt.tight_layout()
plt.show()


# group the dataframe by cluster and data class, and count the number of occurrences
data_class_counts = df.groupby(['Cluster', 'DataClasses']).size().reset_index(name='Count')

# sort the values by cluster and count in descending order
data_class_counts = data_class_counts.sort_values(['Cluster', 'Count'], ascending=[True, False])

# print the top 3 data classes in each cluster
for i in range(n_clusters):
    print(f"Cluster {i}:")
    print(data_class_counts[data_class_counts['Cluster'] == i].head(3))
    print("\n")