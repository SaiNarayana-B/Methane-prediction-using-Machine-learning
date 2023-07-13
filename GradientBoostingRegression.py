from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

X = data[['Temperature', 'pH', 'Pressure']].values
y = data['Methane'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)

y_pred = gb_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

tem = float(input("Temperature: "))
ph = float(input("pH: "))
prs = float(input("Pressure: "))

new_sample = pd.DataFrame([[tem, ph, prs]], columns=['Temperature', 'pH', 'Pressure']).values

prediction = gb_reg.predict(new_sample)

print("Predicted methane gas production:", prediction)
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
