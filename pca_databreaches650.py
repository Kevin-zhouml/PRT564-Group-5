from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

# Read dataset into a DataFrame
df = pd.read_csv("final5.csv")

# Separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,1:].values
y = df.iloc[:,0].values
# apply Standardisation on explanatory variables in training set
std_scaler = preprocessing.StandardScaler()
x_std = std_scaler.fit_transform(x)

# project the dataset onto 2-dimensional subspace
pca = PCA(n_components=5)  # project from 16 to 5 dimensions
projected_2 = pca.fit_transform(x_std)
print(df.shape)
print(projected_2)

#visualise dataset over 5 PCA components
plt.scatter(projected_2[:, 0], projected_2[:, 1],
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()

# compare the cumulative explained variance versus number of PCA components
pca = PCA().fit(x_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# find number of PCA components that explain 90% of the variance
pca = PCA(0.90).fit(x_std)
print("90%% variance is explained by: %.d components." % pca.n_components_)

# find number of PCA components that explain 95% of the variance
pca = PCA(0.95).fit(x_std)
print("95%% variance is explained by: %.d components." % pca.n_components_)

# find number of PCA components that explain 99% of the variance
pca = PCA(0.99).fit(x_std)
print("99%% variance is explained by: %.d components." % pca.n_components_)
