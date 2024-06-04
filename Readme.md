# Linear Regression: Single Variable Analysis

This project demonstrates a simple implementation of linear regression using a single variable to predict per capita income based on historical data.

## Output

## Project Overview

The objective of this project is to use linear regression to predict the per capita income of Canada in 2024 based on historical data. The analysis is performed using Python and several libraries, including pandas, numpy, matplotlib, and scikit-learn.

## Dataset

The dataset used in this analysis is `canada_per_capita_income.csv`, which contains the following columns:
- `year`: The year of the record.
- `per capita income (US$)`: The per capita income in USD for that year.

## Requirements

To run this project, you need to have the following libraries installed:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Code Explanation

### Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
```

### Reading the CSV File

```python
df = pd.read_csv('/content/canada_per_capita_income.csv')
df.info()
df.head()
df.tail()
```

### Plotting Scatter Plot for the Available Data

```python
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Per Capita Income Over Years')

plt.scatter(x=df['year'], y=df['per capita income (US$)'], color='red', marker='+')
```

### Training the Regression Model

```python
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df[['per capita income (US$)']])
```

### Making Predictions

```python
reg.predict([[2024]])
reg.coef_
reg.intercept_
```

### Plotting Linear Regression

```python
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Per Capita Income Over Years')

plt.scatter(x=df['year'], y=df['per capita income (US$)'], color='red', marker='+')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')
```

## Results

The regression model provides the coefficients and intercept, which can be used to predict the per capita income for any given year. In this project, the per capita income for the year 2024 is predicted using the trained model.

## Conclusion

This project provides a basic introduction to linear regression using a single variable. The steps include reading the data, plotting it, training a linear regression model, making predictions, and visualizing the results.

