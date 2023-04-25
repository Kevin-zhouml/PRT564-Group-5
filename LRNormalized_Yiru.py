import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
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

# split the data into training and testing sets
X = df_monthly[['BreachDate']] # input feature
y = df_monthly[['PwnCount']] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# apply min-max scaling to the target variable in the training set
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train)

# apply the same transformation to the testing set
y_test_scaled = scaler.transform(y_test)

# convert 'BreachDate' column to Unix timestamp (numerical format)
X_train_unix = X_train['BreachDate'].apply(lambda x: x.timestamp())
X_test_unix = X_test['BreachDate'].apply(lambda x: x.timestamp())

# create a linear regression model
model = LinearRegression()

# fit the model on the training data
model.fit(X_train_unix.to_frame(), y_train_scaled)

# use the trained model to make predictions on the testing data
y_pred_scaled = model.predict(X_test_unix.to_frame())

# undo the min-max scaling on the predicted values
y_pred = scaler.inverse_transform(y_pred_scaled)

# print the coefficient of determination (R-squared)
print('R-squared:', model.score(X_test_unix.to_frame(), y_test_scaled))

# plot the training data
plt.scatter(X_train_unix, y_train_scaled, color='blue')

# plot the testing data
plt.scatter(X_test_unix, y_test_scaled, color='green')

# plot the predicted line
plt.plot(X_test_unix, y_pred_scaled, color='red')

# set the x-axis label
plt.xlabel('Breach Date')

# set the y-axis label
plt.ylabel('Number of Breached Accounts (MinMax Scaled)')

# set the plot title
plt.title('Linear Regression Prediction of Monthly Breach Count')

# display the plot
plt.show()
