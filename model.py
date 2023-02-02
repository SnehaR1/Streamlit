import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"C:\Users\rasaa\OneDrive\Desktop\streamlit\streamlitenv\pima-indians-diabetes.csv", names=[
                   'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class'])

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, n_features=6)
classifier.fit(X_train, y_train)
ypred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, ypred)
print("Accuracy: ", accuracy)
joblib.dump(classifier, "rf_model.sav")
