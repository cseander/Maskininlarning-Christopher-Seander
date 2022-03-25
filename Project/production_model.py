import pandas as pd
import joblib

samples = pd.read_csv("test_samples.csv")
X, y = samples.drop(["cardio", "Unnamed: 0"], axis = 1), samples["cardio"]

model = joblib.load("model.pkl")
probabolity = model.predict_proba(X)
prediction = model.predict(X)

results = pd.DataFrame([probabolity[:,0], probabolity[:,1], prediction]).T
results.columns = ["probability class 0", "probability class 1", "prediction"]
results.to_csv("prediction.csv")