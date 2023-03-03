import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib


new_csv = pd.read_csv('iris.csv')
# print(new_csv.shape)
# print(new_csv.columns)

new_csv.dropna() # cleaning data from none given values
new_csv.drop_duplicates(inplace=True) # cleaning the data from duplicates

X = new_csv.drop('class', axis=1).values
y = new_csv['class']
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # splitting the data for training and testing parts
# print(X_train, X_test, y_train, y_test)


model = DecisionTreeClassifier() # creating model
model.fit(X_train, y_train) # giving the training data
predict = model.predict(X_test) # given the part of data which the model should learn to use than to predict
score = accuracy_score(y_test, predict) # calculates the percentage of prediction accuracy
print(score)

joblib.dump(model, 'result.joblib') # loads data into joblib


