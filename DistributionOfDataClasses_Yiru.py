import pandas as pd
import matplotlib.pyplot as plt

# read the merged CSV file into a pandas dataframe
df = pd.read_csv('databreaches650.csv')

# create a new DataFrame that contains the count of breaches for each unique DataClasses
data_class_counts = df['DataClasses'].str.replace('[','').str.replace(']','').str.split(',', expand=True) \
    .stack().str.strip().value_counts().reset_index()

# rename columns
data_class_counts.columns = ['DataClasses', 'Count']

print(data_class_counts)

# sort values by count in descending order
data_class_counts = data_class_counts.sort_values('Count', ascending=False)
 
 

# plot the distribution using a bar chart
plt.figure(figsize=(10,6))
plt.bar(data_class_counts['DataClasses'], data_class_counts['Count'])
plt.xticks(fontsize=6, rotation=90)
plt.xlabel('DataClasses')
plt.ylabel('Number of breaches')
plt.title('Distribution of number of breaches based on DataClasses')
plt.tight_layout()
plt.show()
