import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data[['Temperature', 'pH', 'Pressure']].values
y = data['Methane'].values

model = LinearRegression()

model.fit(X, y)

temperature = float(input("Enter the temperature: "))
pH = float(input("Enter the pH: "))
pressure = float(input("Enter the pressure: "))


input_data = np.array([[temperature, pH,pressure]])
prediction = model.predict(input_data)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

print("Predicted methane production:", prediction[0])
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.plot(kind='line')
plt.xlabel('Index')
plt.ylabel('Methane')
plt.title('Actual vs Predicted Methane Gas Production')
plt.show()

metric_names = ['MSE', 'RMSE', 'MAE', 'R-squared']
metric_values = [mse, rmse, mae, r2]


plt.bar(metric_names, metric_values)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of Evaluation Metrics')
plt.show()
