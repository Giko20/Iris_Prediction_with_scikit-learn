import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# df = pd.read_csv('dataset.csv')
# print(df)
# print(df.head())
# new_df = df[['danceability','energy', 'loudness', 'mode', 'speechiness', 'liveness', 'tempo', 'track_genre']].copy()
# print(type(new_df))
# new_df.to_csv('data.csv')

new_csv = pd.read_csv('iris.csv')
# print(new_csv.shape)
# print(new_csv.columns)
# dict = df2.to_dict()

new_csv.dropna()
new_csv.drop_duplicates(inplace=True)
#
# df2 = new_csv.groupby(['track_genre'])['track_genre'].count()
# print(df2.head())
# df3 = df2.head()
# # print(df2)
# # plt.show()
#
# dict2 = df3.to_dict()
# # print(dict.values())
# x = np.array(list(dict2.keys()))
# y = np.array(list(dict2.values()))
#
# plt.bar(x, y)
# plt.show()


X = new_csv.drop('class', axis=1).values
y = new_csv['class']
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# print(X_train, X_test, y_train, y_test)


model = DecisionTreeClassifier()
model.fit(X_train, y_train) # აქ ვაწვდით სატრენინგო მონაცემებს
predict = model.predict(X_test) # აქ რა უნდა იწინასწარმეტყველოს
score = accuracy_score(y_test, predict) # აქ საზღვრავს რამდენი პროცენთით ზუსტად იწინასწარმეტყველებს სწორ შედეგს
print(score)

joblib.dump(model, 'result.joblib')


