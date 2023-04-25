import pandas as pd 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

# read a remote .csv file
df = pd.read_csv('final5.csv')

# separate response variable (y) from explanatory variables (X)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# create training and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# apply Standardisation on explanatory variables in training set
std_scaler = preprocessing.StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)

# initialise 2-component PCA
pca = PCA(n_components=2)

# PCA on original explanatory variables
X_train = pca.fit_transform(X_train)

# PCA on standardised explanatory variables
X_train_std = pca.fit_transform(X_train_std)


# initial a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

# subplot 1 (original) containing 10 scatterplots for 10 class labels of industries
# 1st principal component (x-axis), 2nd principal component (y-axis) 
for l,c,m in zip(range(1,11), ('blue', 'red', 'green', 'purple', 'orange', 'yellow', 'grey', 'black', 'pink', 'brown'), ('^', 's', 'o', '^', 's', 'o', '^', 's', 'o', '^')):
    ax1.scatter(X_train[y_train==l, 0], X_train[y_train==l, 1],
        color=c,
        label='class %s' %l,
        alpha=0.5,
        marker=m
        )

# subplot 2 (standardised) containing 10 scatterplots for 10 class labels of industries
# 1st principal component (x-axis), 2nd principal component (y-axis) 
for l,c,m in zip(range(1,11), ('blue', 'red', 'green', 'purple', 'orange', 'yellow', 'grey', 'black', 'pink', 'brown'), ('^', 's', 'o', '^', 's', 'o', '^', 's', 'o', '^')):
    ax2.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1],
        color=c,
        label='class %s' %l,
        alpha=0.5,
        marker=m
        )
    
# set titles
ax1.set_title('Original training dataset after PCA')    
ax2.set_title('Standardised training dataset after PCA')    

# other settings
for ax in (ax1, ax2):
    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()

plt.show()