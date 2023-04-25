import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# read the merged CSV file into a pandas dataframe
df = pd.read_csv('databreaches650.csv')
print(df)

# convert 'BreachDate' column to datetime object
df['BreachDate'] = pd.to_datetime(df['BreachDate'])

# set the index of the dataframe to 'BreachDate'
df.set_index('BreachDate', inplace=True)

# sort the dataframe by the index in ascending order
df.sort_index(inplace=True)

# resample the dataframe to monthly intervals and sum the 'PwnCount' values
df_monthly = df.resample('M').sum()

# reset the index of the resampled dataframe
df_monthly.reset_index(inplace=True)

# convert 'BreachDate' column to Unix timestamp (numerical format)
df_monthly['BreachDate'] = df_monthly['BreachDate'].apply(lambda x: x.timestamp())

# split the data into training and testing sets
X = df_monthly[['BreachDate']] # input feature
y = df_monthly[['PwnCount']] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a linear regression model
model = LinearRegression()

# fit the model on the training data
model.fit(X_train, y_train)

# use the trained model to make predictions on the testing data
y_pred = model.predict(X_test)

# print the coefficient of determination (R-squared)
print('R-squared:', model.score(X_test, y_test))

# plot the training data
plt.scatter(X_train, y_train, color='blue')

# plot the testing data
plt.scatter(X_test, y_test, color='green')

# plot the predicted line
plt.plot(X_test, y_pred, color='red')

# set the x-axis label
plt.xlabel('Breach Date')

# set the y-axis label
plt.ylabel('Number of Breached Accounts')

# set the plot title
plt.title('Linear Regression Prediction of Monthly Breach Count')

# display the plot
plt.show()
