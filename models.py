# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('breast_cancer.csv')

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
# Drop the column id
df=df.iloc[:,1:]

df[df['bare_nucleoli']=='?'].sum()
digits_in_bare_nucleoli=pd.DataFrame(df.bare_nucleoli.str.isdigit())
data=df.replace("?", np.nan)

X = df.drop('class', axis=1)
y = df['class']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale your data
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler model and fit it to the training data
X_minmax = MinMaxScaler().fit(X_train)

# Transform the training and testing data using the X_scaler and y_scaler models
X_train_minmax = X_minmax.transform(X_train)
X_test_minmax = X_minmax.transform(X_test)

# Train the model
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train_minmax, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Create a random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train, y_train)
rf.score(X_train, y_train)

# Save the model as a pickle in a file 
joblib.dump(rf, 'model.pkl', 'wb') 

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1.8]]))
