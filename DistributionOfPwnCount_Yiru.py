import pandas as pd
import matplotlib.pyplot as plt

# read the merged CSV file into a pandas dataframe
df = pd.read_csv('databreaches650.csv')

# plot a histogram of PwnCount
plt.hist(df['PwnCount'], bins=50)

# set the x-axis label
plt.xlabel('PwnCount')

# set the y-axis label
plt.ylabel('Frequency')

# set the plot title
plt.title('Distribution of PwnCount')

# add the counts on top of each bar
for i in range(len(plt.hist(df['PwnCount'], bins=50)[0])):
    count = plt.hist(df['PwnCount'], bins=50)[0][i]
    plt.text(plt.hist(df['PwnCount'], bins=50)[1][i], count+10, str(int(count)))

# display the plot
plt.show()
