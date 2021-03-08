# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import requests
import json
import joblib

# Importing the dataset
df = pd.read_csv('Data/breast_cancer.csv')

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
# Drop the column id
df=df.iloc[:,1:]

X = df.drop('class', axis=1)
y = df['class']

df = X.copy()

# get_dummies
df_binary_encoded = pd.get_dummies(df, columns=["bare_nucleoli"])
df_binary_encoded.head()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train.head()

# Scale your data
from sklearn.preprocessing import MinMaxScaler
X_minmax = MinMaxScaler().fit(X_train)

# Transform the training and testing data using the X_scaler and y_scaler models
X_train_minmax = X_minmax.transform(X_train)
X_test_minmax = X_minmax.transform(X_test)

# Train the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train_minmax, y_train)

# Predicting the Test set results
predictions = rf.predict(X_test)

# Save the model as a pickle in a file 
joblib.dump(rf, 'model.pkl') 

# Loading model to compare the results
model = pickle.load(open('model.pkl'))
print(model.predict([[1.8]]))
