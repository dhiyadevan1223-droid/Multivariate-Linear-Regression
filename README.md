# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:


1. **Load and Prepare Dataset**
   Import required libraries, load the California housing dataset, and split the data into training and testing sets.

2. **Create and Train Model**
   Create a Linear Regression model and train it using the training dataset.

3. **Evaluate the Model**
   Predict the output values, display regression coefficients, and calculate the variance score (R²).

4. **Plot Residual Errors**
   Plot the residual errors for both training and testing data, draw the zero-error line, add legend/title, and display the graph.

## Program:
```

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Load California housing dataset
housing = fetch_california_housing()
X = housing.data      # feature matrix
y = housing.target    # target vector
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
# Create linear regression object
reg = linear_model.LinearRegression()
# Train the model
reg.fit(X_train, y_train)
# Regression coefficients
print('Coefficients:', reg.coef_)
# Variance score (R^2)
print('Variance score:', reg.score(X_test, y_test))
# Plot residual errors
plt.style.use('fivethirtyeight')
# Residuals for training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,color="green", s=10, label='Train data')
# Residuals for testing data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,color="blue", s=10, label='Test data')
# Zero residual line
plt.hlines(y=0, xmin=min(reg.predict(X_train)), xmax=max(reg.predict(X_train)),linewidth=2)
# Legend and title
plt.legend(loc='upper right')
plt.title("Residual errors")
# Show plot
plt.show()

```
## Output:

<img width="1000" height="760" alt="image" src="https://github.com/user-attachments/assets/2dd9212e-7c16-470b-9cc2-8576955df3b3" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
