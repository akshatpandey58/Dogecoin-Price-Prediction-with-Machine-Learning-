# Cryptocurrency Data Analysis and Prediction

This project involves analyzing historical cryptocurrency data and making predictions using statistical models. The dataset used in this project is specifically for Dogecoin (DOGE-USD).

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

Ensure you have these libraries installed. You can install them using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Dataset

The dataset used in this project is a CSV file containing historical data for Dogecoin (DOGE-USD). The file should have columns including 'Date', 'Close', 'Volume', 'High', 'Low'.

## Steps

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
```

### 2. Load Data

```python
data = pd.read_csv('path_to_your_csv_file.csv')
data.head()
```

### 3. Data Preprocessing

- Convert 'Date' column to datetime format.
- Set 'Date' as the index.
- Check for and drop any missing values.

```python
data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.set_index('Date', inplace=True)
data = data.dropna()
```

### 4. Data Analysis

- Compute correlation matrix.
- Create new features for the analysis.

```python
data['gap'] = (data['High'] - data['Low']) * data['Volume']
data['y'] = data['High'] / data['Volume']
data['z'] = data['Low'] / data['Volume']
data['a'] = data['High'] / data['Low']
data['b'] = (data['High'] / data['Low']) * data['Volume']
abs(data.corr()['Close'].sort_values(ascending=False))
```

### 5. Data Visualization

Plot the 'Close' prices over time.

```python
plt.figure(figsize=(20, 7))
x = data.groupby('Date')['Close'].mean()
x.plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Close of 2021")
```

### 6. Model Training

Split the data into training and testing sets. Train a SARIMAX model on the training data.

```python
df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

model = SARIMAX(endog=train['Close'], exog=train.drop('Close', axis=1), order=(2, 1, 1))
results = model.fit()
print(results.summary())
```

### 7. Predictions

Make predictions using the trained model.

```python
start = 11
end = 29
predictions = results.predict(start=start, end=end, exog=test.drop('Close', axis=1))

test['Close'].plot(legend=True, figsize=(12, 6))
predictions.plot(label='TimeSeries', legend=True)
```

## Conclusion

This project demonstrates how to analyze historical cryptocurrency data and make predictions using statistical models. The SARIMAX model provides a method for making time series predictions with exogenous variables.

