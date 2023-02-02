import joblib


def predict(data):
    classifier = joblib.load("rf_model.sav")
    return classifier.predict(data)
