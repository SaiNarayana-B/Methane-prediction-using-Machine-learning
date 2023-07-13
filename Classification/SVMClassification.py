from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data[['Temperature', 'pH', 'Pressure']].values
y = data['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')

svm_classifier.fit(X_train_scaled, y_train)

y_pred = svm_classifier.predict(X_test_scaled)

temperature = float(input("Temperature: "))
ph = float(input("pH: "))
pressure = float(input("Pressure: "))

new_sample = [[temperature, ph, pressure]]
new_sample_scaled = scaler.transform(new_sample)
prediction = svm_classifier.predict(new_sample_scaled)

print("Predicted category:", prediction)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100,'%')
