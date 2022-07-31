#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv("trainDL.csv")


# In[3]:


# Configuring float numbers format
pd.options.display.float_format = '{:20.2f}'.format
dataset.head(n=5)


# In[5]:


dataset.describe(include=[np.number], percentiles=[.5]) .transpose().drop("count", axis=1)


# In[6]:


#we move to see statistical information about the non-numerical columns in our dataset:
dataset.describe(include=[np.object]).transpose() .drop("count", axis=1)


# In[ ]:


#Dealing with Missing Values


# In[7]:


# Getting the number of missing values in each column
num_missing = dataset.isna().sum()
# Excluding columns that contains 0 missing values
num_missing = num_missing[num_missing > 0]
# Getting the percentages of missing values
percent_missing = num_missing * 100 / dataset.shape[0]


# In[8]:


# Concatenating the number and perecentage of missing values
# into one dataframe and sorting it
pd.concat([num_missing, percent_missing], axis=1,
keys=['Missing Values', 'Percentage']).\
sort_values(by="Missing Values", ascending=False)


# In[11]:


dataset["PoolArea"].value_counts()


# In[13]:


dataset["PoolQC"].fillna("No Pool", inplace=True)


# In[15]:


dataset["MiscVal"].value_counts()


# In[17]:


dataset['MiscFeature'].fillna('No feature', inplace=True)


# In[19]:


dataset['Alley'].fillna('No Alley', inplace=True)
dataset['Fence'].fillna('No Fence', inplace=True)
dataset['FireplaceQu'].fillna('No Fireplace', inplace=True)


# In[20]:


dataset['LotFrontage'].fillna(0, inplace=True)


# In[31]:


garage_columns = [col for col in dataset.columns if col.startswith("Garage")]
dataset[dataset['GarageCars'].isna()][garage_columns]


# In[32]:


dataset[~pd.isna(dataset['GarageType']) &
pd.isna(dataset['GarageQual'])][garage_columns]


# In[33]:


dataset['GarageCars'].fillna(0, inplace=True)
dataset['GarageArea'].fillna(0, inplace=True)


# In[34]:


dataset.loc[~pd.isna(dataset['GarageType']) &
pd.isna(dataset['GarageQual']), "GarageType"] = "No Garage"


# In[36]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
dataset[col].fillna('NoGarage', inplace=True)


# In[38]:


dataset[~pd.isna(dataset['GarageType']) &
pd.isna(dataset['GarageQual'])][garage_columns]


# In[39]:


dataset['GarageCars'].fillna(0, inplace=True)
dataset['GarageArea'].fillna(0, inplace=True)


# In[40]:


dataset.loc[~pd.isna(dataset['GarageType']) &
pd.isna(dataset['GarageQual']), "GarageType"] = "No Garage"


# In[124]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    dataset[col].fillna('NoGarage', inplace=True)


# In[125]:


dataset['GarageYrBlt'].fillna(0, inplace=True)


# In[128]:


bsmt_columns = [col for col in dataset.columns if "Bsmt" in col]
dataset[dataset['BsmtHalfBath'].isna()][bsmt_columns]


# In[131]:


dataset[~pd.isna(dataset['BsmtCond']) &
pd.isna(dataset['BsmtExposure'])][bsmt_columns]


# In[45]:


dataset[~pd.isna(dataset['BsmtCond']) &
pd.isna(dataset['BsmtFinType2'])][bsmt_columns]


# In[113]:


for col in ["BsmtHalfBath", "BsmtFullBath", "TotalBsmtSF","BsmtUnfSF", "BsmtFinSF2", "BsmtFinSF1"]:
    dataset[col].fillna(0,inplace=True)


# In[114]:


dataset.loc[~pd.isna(dataset['BsmtCond']) &
pd.isna(dataset['BsmtExposure']), "BsmtExposure"] = "No"
dataset.loc[~pd.isna(dataset['BsmtCond']) &
pd.isna(dataset['BsmtFinType2']), "BsmtFinType2"] = "Unf"


# In[115]:


for col in ["BsmtExposure", "BsmtFinType2","BsmtFinType1", "BsmtQual", "BsmtCond"]:
    dataset[col].fillna("NoBasement", inplace=True)


# In[116]:


dataset['MasVnrArea'].fillna(0, inplace=True)
dataset['MasVnrType'].fillna("None", inplace=True)


# In[117]:


dataset['Electrical'].fillna(dataset['Electrical'].mode()[0], inplace=True)


# In[118]:


dataset.isna().values.sum()


# In[119]:


null = dataset.isnull().sum()
null = null[null.values>0]


# In[120]:


null.sort_values(ascending=False)


# In[121]:


#Data cleaning and preprocessing
#Some of the columns have 'NA' as a category of the values.
col_with_NA_as_category = ['GarageType','GarageFinish','GarageQual','GarageCond']


# In[122]:


#We should remove the null values only for those columns which does not have 'NA' as a category of values.
null_col = [i for i in null.index if i not in col_with_NA_as_category]
null_col


# In[123]:


dataset.isna().values.sum()


# In[132]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    dataset[col].fillna('NoGarage', inplace=True)


# In[133]:


dataset.isna().values.sum()


# In[ ]:


#Outlier Removal


# In[134]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[135]:


plt.scatter(x=dataset['GrLivArea'], y=dataset['SalePrice'],
            color="orange", edgecolors="#000000", linewidths=0.5);
plt.xlabel("GrLivArea"); plt.ylabel("SalePrice");


# In[137]:


outlirt_columns = ["GrLivArea"] +                   [col for col in dataset.columns if "Sale" in col]
dataset[dataset["GrLivArea"] > 4000][outlirt_columns]


# In[138]:


#remove Outliers:
dataset = dataset[dataset["GrLivArea"] < 4000]
plt.scatter(x=dataset['GrLivArea'], y=dataset['SalePrice'],
            color="orange", edgecolors="#000000", linewidths=0.5);
plt.xlabel("GrLivArea"); plt.ylabel("SalePrice");


# In[139]:


dataset.reset_index(drop=True, inplace=True)


# In[ ]:


dataset.drop(['Id'], axis=1, inplace=True)


# Exploratory Data Analysis

# In[ ]:


#Target Variable Distribution


# In[141]:


sns.violinplot(x=dataset['SalePrice'], inner="quartile", color="#36B37E");


# In[142]:


sns.boxplot(dataset['SalePrice'], whis=10, color="#00B8D9");


# In[143]:


sns.distplot(dataset['SalePrice'], kde=False,
color="#172B4D", hist_kws={"alpha": 0.8});
plt.ylabel("Count");


# Correlation Between Variables

# In[144]:


fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(dataset.corr(), ax=ax);


# In[145]:


#Relatioships Between the Target Variable and Other Varibles


# In[146]:


sns.distplot(dataset['SalePrice'], kde=False,
color="#172B4D", hist_kws={"alpha": 0.8});
plt.ylabel("Count");


# In[147]:


sns.distplot(dataset['OverallQual'], kde=False,
color="#172B4D", hist_kws={"alpha": 1});
plt.ylabel("Count");


# In[148]:


plt.scatter(x=dataset['OverallQual'], y=dataset['SalePrice'],
color="orange", edgecolors="#000000", linewidths=0.5);
plt.xlabel("OverallQual"); plt.ylabel("SalePrice");


# In[149]:


sns.distplot(dataset['GrLivArea'], kde=False,
color="#172B4D", hist_kws={"alpha": 0.8});
plt.ylabel("Count");


# In[150]:


plt.scatter(x=dataset['GrLivArea'], y=dataset['SalePrice'],
color="red", edgecolors="#000000", linewidths=0.5);
plt.xlabel("GrLivArea"); plt.ylabel("SalePrice");


# In[ ]:


#Moderate Positive Correlation


# In[151]:


fig, axes = plt.subplots(1, 4, figsize=(18,5))
fig.subplots_adjust(hspace=0.5, wspace=0.6)
for ax, v in zip(axes.flat, ["YearBuilt", "YearRemodAdd","MasVnrArea", "TotalBsmtSF"]):
    sns.distplot(dataset[v], kde=False, color="#172B4D",
                 hist_kws={"alpha": 0.8}, ax=ax)
ax.set(ylabel="Count");


# In[ ]:


#relationships with the target variable using scatter plots:


# In[152]:


x_vars = ["YearBuilt", "YearRemodAdd", "MasVnrArea", "TotalBsmtSF"]
g = sns.PairGrid(dataset, y_vars=["SalePrice"], x_vars=x_vars);
g.map(plt.scatter, color="orange", edgecolors="#000000", linewidths=0.5);


# In[ ]:


#e the distribution of each


# In[154]:


fig, axes = plt.subplots(1, 4, figsize=(18,5))
fig.subplots_adjust(hspace=0.5, wspace=0.6)
for ax, v in zip(axes.flat, ["1stFlrSF", "FullBath", "GarageCars", "GarageArea"]):
    sns.distplot(dataset[v], kde=False, color="#172B4D",hist_kws={"alpha": 0.8}, ax=ax);
    ax.set(ylabel="Count");


# In[ ]:


# their relationships with the target variable:


# In[155]:


x_vars = ["1stFlrSF", "FullBath", "GarageCars", "GarageArea"]
g = sns.PairGrid(dataset, y_vars=["SalePrice"], x_vars=x_vars);
g.map(plt.scatter, color="orange", edgecolors="#000000", linewidths=0.5);


# In[ ]:


#Relatioships Between Predictor Variables


# In[157]:


sns.distplot(dataset['TotRmsAbvGrd'], kde=False,color="#172B4D", hist_kws={"alpha": 0.8});
plt.ylabel("Count");


# In[158]:


plt.rc("grid", linewidth=0.05)
fig, axes = plt.subplots(1, 2, figsize=(15,5))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
h1 = axes[0].hist2d(dataset["GarageCars"],
                    dataset["GarageArea"],
                    cmap="viridis");
axes[0].set(xlabel="GarageCars", ylabel="GarageArea")
plt.colorbar(h1[3], ax=axes[0]);
h2 = axes[1].hist2d(dataset["GrLivArea"],
                    dataset["TotRmsAbvGrd"],
                    cmap="viridis");
axes[1].set(xlabel="GrLivArea", ylabel="TotRms AbvGrd")
plt.colorbar(h1[3], ax=axes[1]);
plt.rc("grid", linewidth=0.25)


# In[ ]:


#Negative Correlation


# In[160]:


fig, axes = plt.subplots(1, 3, figsize=(16,5))
fig.subplots_adjust(hspace=0.5, wspace=0.6)
for ax, v in zip(axes.flat, ["BsmtUnfSF", "BsmtFinSF1", "BsmtFullBath"]):
    sns.distplot(dataset[v], kde=False, color="#172B4D",
                 hist_kws={"alpha": 0.8}, ax=ax);
    ax.set(ylabel="Count")


# In[ ]:


#the relationship between each pair


# In[161]:


fig, axes = plt.subplots(1, 2, figsize=(15,5))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
axes[0].scatter(dataset["BsmtUnfSF"], dataset["BsmtFinSF1"],
                color="red", edgecolors="#000000", linewidths=0.5);
axes[0].set(xlabel="BsmtUnfSF", ylabel="BsmtFinSF1");
axes[1].scatter(dataset["BsmtUnfSF"], dataset["BsmtFullBath"],
                color="red", edgecolors="#000000", linewidths=0.5);
axes[1].set(xlabel="BsmtUnfSF", ylabel="BsmtFullBath");


# In[164]:


#Feature Engineering 
#Creating New Derived Features


# In[165]:


for f in ["OverallQual", "GrLivArea"]:
    dataset[f + "_p2"] = dataset[f] ** 2
    dataset[f + "_p3"] = dataset[f] ** 3
dataset["OverallQual_GrLivArea"] =     dataset["OverallQual"] * dataset["GrLivArea"]


# In[166]:


dataset.drop(["GarageCars", "TotRmsAbvGrd"], axis=1, inplace=True)


# In[167]:


print("Unique values in 'BsmtCond' column:")
print(dataset['BsmtCond'].unique().tolist())


# In[168]:


mp = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
dataset['ExterQual'] = dataset['ExterQual'].map(mp)
dataset['ExterCond'] = dataset['ExterCond'].map(mp)
dataset['HeatingQC'] = dataset['HeatingQC'].map(mp)
dataset['KitchenQual'] = dataset['KitchenQual'].map(mp)


# In[170]:


mp = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Basement':0}
dataset['BsmtQual'] = dataset['BsmtQual'].map(mp)
dataset['BsmtCond'] = dataset['BsmtCond'].map(mp)
dataset['BsmtExposure'] = dataset['BsmtExposure'].map(
    {'Gd':4,'Av':3,'Mn':2,'No':1,'No Basement':0})


# In[172]:


mp = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'No Basement':0}
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map(mp)
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map(mp)


# In[173]:


dataset['CentralAir'] = dataset['CentralAir'].map({'Y':1,'N':0})


# In[174]:


dataset['Functional'] = dataset['Functional'].map(
    {'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,
    'Maj2':2,'Sev':1,'Sal':0})
dataset['FireplaceQu'] = dataset['FireplaceQu'].map(
    {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Fireplace':0})
dataset['GarageFinish'] = dataset['GarageFinish'].map(
    {'Fin':3,'RFn':2,'Unf':1,'No Garage':0})
dataset['GarageQual'] = dataset['GarageQual'].map(
    {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
dataset['GarageCond'] = dataset['GarageCond'].map(
    {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No Garage':0})
dataset['PoolQC'] = dataset['PoolQC'].map(
    {'Ex':4,'Gd':3,'TA':2,'Fa':1,'No Pool':0})
dataset['LandSlope'] = dataset['LandSlope'].map(
    {'Sev': 2, 'Mod': 1, 'Gtl': 0})
dataset['Fence'] = dataset['Fence'].map(
    {'GdPrv':4,'MnPrv':3,'GdWo':2,'MnWw':1,'No Fence':0})


# In[ ]:


#One-Hot Encoding For Categorical Features


# In[175]:


dataset[['PavedDrive']].head()


# In[176]:


dataset = pd.get_dummies(dataset)


# In[177]:


pavedDrive_oneHot = [c for c in dataset.columns if c.startswith("Paved")]
dataset[pavedDrive_oneHot].head()


# In[ ]:


#Prediction Type and Modeling Techniques


# In[178]:


dataset[['SalePrice']].head()


# In[ ]:


# Linear Regression


# In[179]:


from sklearn.preprocessing import StandardScaler


# In[180]:


scaler = StandardScaler()
# We need to fit the scaler to our data before transformation
dataset.loc[:, dataset.columns != 'SalePrice'] = scaler.fit_transform(
    dataset.loc[:, dataset.columns != 'SalePrice'])


# In[181]:


from sklearn.model_selection import train_test_split


# In[182]:


X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop('SalePrice', axis=1), dataset[['SalePrice']],
    test_size=0.25, random_state=3)


# In[ ]:


#Searching for Effective Parameters


# In[187]:


from sklearn.tree import DecisionTreeRegressor


# In[188]:


model = DecisionTreeRegressor(max_depth=14, min_samples_split=5, max_features=20)


# In[190]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameter_space = {
    "max_depth": [7, 15],
    "min_samples_split": [5, 10],
    "max_features": [30, 45]
}
clf = GridSearchCV(DecisionTreeRegressor(), parameter_space, cv=4,
                scoring="neg_mean_absolute_error")
clf.fit(X_train, y_train)


# In[192]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV


# In[193]:


clf = RandomizedSearchCV(DecisionTreeRegressor(), parameter_space, cv=4,
                        scoring="neg_mean_absolute_error", n_iter=100)


# In[ ]:


#Performance Metric


# In[ ]:


#Modeling 
#Linear Regression  
#Ridge Regression 


# In[199]:


from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


# In[ ]:


Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
      max_iter=None, tol=0.001, solver=auto, random_state=None)


# In[ ]:


parameter_space = {
    "alpha": [1, 10, 100, 290, 500],
    "fit_intercept": [True, False],
    "solver": ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
}
clf = GridSearchCV(Ridge(random_state=3), parameter_space, n_jobs=4,
                cv=3, scoring="neg_mean_absolute_error")
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[203]:


ridge_model = Ridge(random_state=3, **clf.best_params_)
ridge_model.fit(X_train, y_train);


# In[ ]:


from sklearn.metrics import mean_absolute_error
y_pred = ridge_model.predict(X_test)
ridge_mae = mean_absolute_error(y_test, y_pred)
print("Ridge MAE =", ridge_mae)


# In[211]:


from sklearn import linear_model
import statsmodels.api as sm
ElasticNet = linear_model.ElasticNet()


# In[ ]:


ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False,
           precompute=False, max_iter=1000, copy_X=True, tol=0.0001,
           warm_start=False, positive=False, random_state=None, selection=cyclic)


# In[ ]:


from sklearn.linear_model import ElasticNet
parameter_space = {
    "alpha": [1, 10, 100, 280, 500],
    "l1_ratio": [0.5, 1],
    "fit_intercept": [True, False],
}
clf = GridSearchCV(ElasticNet(random_state=3), parameter_space,
                   n_jobs=4, cv=3, scoring="neg_mean_absolute_error")
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[214]:


elasticNet_model = ElasticNet(random_state=3, **clf.best_params_)
elasticNet_model.fit(X_train, y_train);


# In[ ]:


y_pred = elasticNet_model.predict(X_test)
elasticNet_mae = mean_absolute_error(y_test, y_pred)
print("Elastic Net MAE =", elasticNet_mae)


# In[ ]:


#Nearest Neighbors
from sklearn import neighbors
KNN_ = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance').fit(X, y)
#KNeighborsRegressor(n_neighbors=5, weights=uniform, algorithm=auto,
leaf_size=30, p=2, metric=minkowski, metric_params=None,
n_jobs=None, **kwargs)


# In[ ]:


parameter_space = {
    "n_neighbors": [9, 10, 11,50],
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": [1,2,20,50,200]
}
clf = GridSearchCV(KNeighborsRegressor(), parameter_space, cv=3,
    scoring="neg_mean_absolute_error", n_jobs=4)
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[220]:


knn_model = KNeighborsRegressor(**clf.best_params_)
knn_model.fit(X_train, y_train);


# In[222]:


y_pred = knn_model.predict(X_test)
knn_mae = mean_absolute_error(y_test, y_pred)
print("K-Nearest Neighbors MAE =", knn_mae)


# In[ ]:


#Support Vector Regression


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR


# In[ ]:


parameter_space = {
        "kernel": ["poly", "linear", "rbf", "sigmoid"],
        "degree": [3, 5],
        "coef0": [0, 3, 7],
        "gamma":[1e-3, 1e-1, 1/X_train.shape[1]],
        "C": [1, 10, 100],
}
clf = GridSearchCV(SVR(), parameter_space, cv=3, n_jobs=4,
                    scoring="neg_mean_absolute_error")
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[225]:


svr_model = SVR(**clf.best_params_)
svr_model.fit(X_train, y_train);
y_pred = svr_model.predict(X_test)
svr_mae = mean_absolute_error(y_test, y_pred)
print("Support Vector Regression MAE =", svr_mae)


# In[ ]:


#Decision Tree


# In[231]:


from sklearn.metrics import mean_squared_error


# In[ ]:


DecisionTreeRegressor(criterion=mse, splitter=best, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0, max_features=None,
                        random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                        min_impurity_split=None, presort=False)


# In[233]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


parameter_space = {
        "criterion": ["mse", "friedman_mse", "mae"],
        "min_samples_split": [5, 18, 29, 50],
        "min_samples_leaf": [3, 7, 15, 25],
        "max_features": [20, 50, 150, 200, X_train.shape[1]],
    }
clf = GridSearchCV(DecisionTreeRegressor(random_state=3), parameter_space,
                    cv=3, scoring="neg_mean_absolute_error", n_jobs=4)
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[235]:


dt_model = DecisionTreeRegressor(**clf.best_params_)
dt_model.fit(X_train, y_train);


# In[ ]:


y_pred = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, y_pred)
print("Decision Tree MAE =", dt_mae)


# In[ ]:





# In[ ]:





# In[ ]:


#Neural Network


# In[239]:


from sklearn.neural_network import MLPRegressor
from network3 import ReLu, linear, ConvPoolLayer


# In[247]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


MLPRegressor(hidden_layer_sizes=(100, ), activation=relu, solver=adam,
            alpha=0.0001, batch_size=auto, learning_rate=constant,
            learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
            random_state=None, tol=0.0001, verbose=False, warm_start=False,
            momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            n_iter_no_change=10)


# In[ ]:


parameter_space = {
        "hidden_layer_sizes": [(7,)*3, (19,), (100,), (154,)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs"],
        "alpha": [1, 10, 100],
}
clf = GridSearchCV(MLPRegressor(random_state=3), parameter_space,
                    cv=3, scoring="neg_mean_absolute_error", n_jobs=4)
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[ ]:


nn_model = MLPRegressor(**clf.best_params_)


# In[ ]:


nn_model.fit(X_train, y_train);


# In[ ]:


y_pred = nn_model.predict(X_test)
nn_mae = mean_absolute_error(y_test, y_pred)
print("Neural Network MAE =", nn_mae)


# In[ ]:


#Random Forest


# In[250]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


parameter_space = {
        "n_estimators": [10, 100, 300, 600],
        "criterion": ["mse", "mae"],
        "max_depth": [7, 50, 254],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 5],
        "max_features": [19, 100, X_train.shape[1]],
        "bootstrap": [True, False],
}
clf = RandomizedSearchCV(RandomForestRegressor(random_state=3),
                        parameter_space, cv=3, n_jobs=4,
                        scoring="neg_mean_absolute_error",
                        n_iter=10, random_state=3)
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[ ]:


rf_model = RandomForestRegressor(**clf.best_params_)


# In[ ]:


rf_model.fit(X_train, y_train);


# In[ ]:


y_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred)
print("Random Forest MAE =", rf_mae)


# In[ ]:


#Gradient Boosting


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


parameter_space = {
        "max_depth": [4, 5, 6],
        "learning_rate": [0.005, 0.009, 0.01],
        "n_estimators": [700, 1000, 2500],
        "booster": ["gbtree",],
        "gamma": [7, 25, 100],
        "subsample": [0.3, 0.6],
        "colsample_bytree": [0.5, 0.7],
        "colsample_bylevel": [0.5, 0.7,],
        "reg_alpha": [1, 10, 33],
        "reg_lambda": [1, 3, 10],
}
clf = RandomizedSearchCV(XGBRegressor(random_state=3),
                        parameter_space, cv=3, n_jobs=4,
                        scoring="neg_mean_absolute_error",
                        random_state=3, n_iter=10)
clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.best_params_)


# In[ ]:


xgb_model = XGBRegressor(**clf.best_params_)


# In[ ]:


xgb_model.fit(X_train, y_train);


# In[ ]:


y_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, y_pred)
print("XGBoost MAE =", xgb_mae)


# In[ ]:


#Analysis and Comparison


# In[ ]:


x = ['KNN', 'Decision Tree', 'Neural Network', 'Ridge',
    'Elastic Net', 'Random Forest', 'SVR', 'XGBoost']
y = [22780.14, 20873.95, 15656.38, 15270.46, 14767.91,
    14506.46, 12874.93, 12556.68]
colors = ["#392834", "#5a3244", "#7e3c4d", "#a1484f",
        "#c05949", "#d86f3d", "#e88b2b", "#edab06"]
fig, ax = plt.subplots()
plt.barh(y=range(len(x)), tick_label=x, width=y, height=0.4, color=colors);
ax.set(xlabel="MAE (smaller is better)", ylabel="Model");


# In[ ]:


#Performance Interpretation


# In[251]:


sns.violinplot(x=dataset['SalePrice'], inner="quartile", color="#36B37E");


# In[252]:


sns.boxplot(dataset['SalePrice'], whis=10, color="#00B8D9");


# In[253]:


sns.distplot(dataset['SalePrice'], kde=False,
            color="#172B4D", hist_kws={"alpha": 0.8});


# In[254]:


y_train.describe(include=[np.number])


# In[ ]:


#Feature Importances XGBoost


# In[ ]:


xgb_feature_importances = xgb_model.feature_importances_
xgb_feature_importances = pd.Series(
    xgb_feature_importances, index=X_train.columns.values
    ).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=xgb_feature_importances,
            y=xgb_feature_importances.index,
            color="#003f5c");
plt.xlabel('Feature Importance');
plt.ylabel('Feature');


# In[ ]:


#Random Forest


# In[ ]:


rf_feature_importances = rf_model.feature_importances_
rf_feature_importances = pd.Series(
    rf_feature_importances, index=X_train.columns.values
    ).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=rf_feature_importances,
            y=rf_feature_importances.index,
            color="#ffa600");
plt.xlabel('Feature Importance');
plt.ylabel('Feature');


# In[ ]:


#Common Important Features


# In[ ]:


common_imp_feat = [x for x in xgb_feature_importances.index
                    if x in rf_feature_importances.index]
commImpFeat_xgb_scores = [xgb_feature_importances[x]
                        for x in common_imp_feat]
commImpFeat_rf_scores = [rf_feature_importances[x]
                        for x in common_imp_feat]
ind = np.arange(len(commImpFeat_xgb_scores))
width = 0.35
fig, ax = plt.subplots()
ax.bar(ind - width/2, commImpFeat_xgb_scores, width,
        color='#003f5c', label='XGBoost');
ax.bar(ind + width/2, commImpFeat_rf_scores, width,
        color='#ffa600', label='Random Forest')
ax.set_xticks(ind);
ax.set_xticklabels(common_imp_feat);
ax.legend();
plt.xticks(rotation=90);


# In[ ]:




