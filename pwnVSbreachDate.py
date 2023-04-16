import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
df = pd.read_csv('PvsD.csv')

# convert the BreachDate column to a datetime format
df['BreachDate'] = pd.to_datetime(df['BreachDate'])

# group the data by year and sum the PwnCount column
yearly_data = df.groupby(df['BreachDate'].dt.year)['PwnCount'].sum()

# Create a scatter plot
plt.scatter(df['BreachDate'], df['PwnCount'])

# Set axis labels and title
plt.xlabel('Breach Date')
plt.ylabel('Pwn Count')
plt.title('Pwn Count vs Breach Date')

# Show the plot
plt.show()
