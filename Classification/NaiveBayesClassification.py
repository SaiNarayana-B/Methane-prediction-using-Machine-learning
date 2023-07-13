import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

extended_data = pd.read_csv('data.csv')

features = extended_data[['Temperature', 'pH', 'Pressure']]
target = extended_data['Category']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)

temperature = float(input("Enter Temperature: "))
ph = float(input("Enter pH: "))
pressure = float(input("Enter Pressure: "))

predicted_category = nb_classifier.predict([[temperature, ph, pressure]])

print("Predicted Category:", predicted_category[0])

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100,'%')
