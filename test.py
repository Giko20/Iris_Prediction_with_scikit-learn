import joblib

model = joblib.load('result.joblib')

prediction = model.predict([[6,2.2,4,1]])
print(prediction)