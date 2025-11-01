# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 
### Name : Mohamed Hameem Sajith J
### Reg: 212223240090
### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
# 1. Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 2. Load the dataset
print("Loading data...")
try:
    # Added on_bad_lines='skip' to handle potential formatting issues in the CSV
    df_full = pd.read_csv('GlobalLandTemperaturesByCity.csv', on_bad_lines='skip')
except FileNotFoundError:
    print("Error: 'GlobalLandTemperaturesByCity.csv' not found.")
    print("Please make sure the file is in the same directory as your script.")
    exit()

# --- Pre-processing the data ---
# Filter for one city to create a single time series
# We will use 'Madras' (Chennai) as our example city
df_city = df_full[df_full['City'] == 'Madras'].copy()

# Convert 'dt' column to datetime objects
df_city['Date'] = pd.to_datetime(df_city['dt'])

# Set the 'Date' as the index
df_city = df_city.set_index('Date')

# Select only the 'AverageTemperature' column and drop missing values
df = df_city[['AverageTemperature']].dropna()

# Resample to 'Monthly Start' frequency and fill gaps
data_monthly = df['AverageTemperature'].resample('MS').mean().ffill()

# Convert Series to DataFrame for consistency
data = data_monthly.to_frame(name='AverageTemperature')

print("First 5 rows of the filtered dataset (Madras):")
print(data.head())
print("\n")

# 3. Plot the time series
print("Displaying Temperature Time Series Plot...")
plt.figure(figsize=(12, 6))
plt.plot(data['AverageTemperature'])
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Temperature Time Series (Madras)')
plt.grid(True)
plt.savefig('exp10_time_series.png')
print("Saved 'exp10_time_series.png'")
plt.show()

# 4. Check for stationarity (using the function from the PDF)
print("--- Stationarity Check (ADF Test) ---")
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.3f}'.format(key, value))

check_stationarity(data['AverageTemperature'])
print("----------------------------------------\n")

# 5. Plot ACF and PACF
print("Displaying ACF and PACF plots...")
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(data['AverageTemperature'], lags=40, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')
ax[0].grid(True)

plot_pacf(data['AverageTemperature'], lags=40, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF)')
ax[1].grid(True)

plt.savefig('exp10_acf_pacf.png')
print("Saved 'exp10_acf_pacf.png'")
plt.show()

# 6. Split the data
train_size = int(len(data) * 0.8)
train, test = data['AverageTemperature'][:train_size], data['AverageTemperature'][train_size:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}\n")

# 7. Fit the SARIMA model
# We use the parameters from the PDF: order=(1,1,1), seasonal_order=(1,1,1,12)
# m=12 because the seasonality is monthly (12 months in a year)
print("Fitting SARIMA model... (This may take a moment)")
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False) # disp=False to suppress convergence output
print("Model fitting complete.\n")

# 8. Make time series predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
# Ensure predictions align with test index for plotting
predictions.index = test.index

# 9. Evaluate model predictions
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'--- Model Evaluation ---')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'--------------------------\n')

# 10. Plot the results
print("Displaying Final Prediction Plot...")
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('SARIMA Model Predictions vs Actual Data')
plt.legend()
plt.grid(True)
plt.savefig('exp10_predictions.png')
print("Saved 'exp10_predictions.png'")
plt.show()

print("Experiment 10 Complete.")
```

### OUTPUT:
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/9f93e709-5e16-4926-a4f9-1ade81be27c2" />

<img width="319" height="140" alt="image" src="https://github.com/user-attachments/assets/793f24ae-26ee-4899-921b-6a0e0058fadc" />

<img width="1312" height="451" alt="image" src="https://github.com/user-attachments/assets/4fcdee93-2957-41c3-a8cf-55c4fedfb3d3" />

<img width="423" height="171" alt="image" src="https://github.com/user-attachments/assets/b799cdd9-4fb0-4b04-8a35-9c3fc9606fc9" />

<img width="999" height="547" alt="image" src="https://github.com/user-attachments/assets/35d3ec31-6487-4527-b161-ef9c271392f3" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
