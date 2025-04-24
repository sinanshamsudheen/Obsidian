### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Linear Regression Single Variable::Linear Regression with a Single Variable is a statistical method that models the relationship between one independent variable (predictor) and one dependent variable (outcome) using a linear equation. It finds the best-fitting straight line (the regression line) through the data points by minimizing the sum of squared differences between observed and predicted values.

---
### Context  
Where and when is it used? Why is it important?

In what context is Linear Regression Single Variable typically applied::Single variable linear regression is applied when analyzing the relationship between two continuous variables, such as predicting house prices based on square footage, sales based on advertising spend, or crop yield based on rainfall. It's important because it provides a simple yet powerful way to quantify relationships, make predictions, and serves as a foundation for more complex regression techniques.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Linear_Regression_Multiple_Variables]]
- [[Gradient_Descent_and_Cost_Function]]
- [[Training_and_Testing_Data]]
- [[L1_and_L2_Regularization]]

What concepts are connected to Linear Regression Single Variable::[[What_is_Machine_Learning]], [[Linear_Regression_Multiple_Variables]], [[Gradient_Descent_and_Cost_Function]], [[Training_and_Testing_Data]], [[L1_and_L2_Regularization]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Linear Regression with Single Variable using scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X = np.array([[5], [15], [25], [35], [45], [55]]) # Hours studied
y = np.array([50, 60, 70, 80, 90, 95])           # Exam scores

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Display results
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Mean squared error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Linear Regression: Hours Studied vs Exam Score')
plt.legend()
plt.show()
```

The equation of the regression line: y = mx + b, where:
- y is the predicted value (dependent variable)
- x is the input value (independent variable)
- m is the coefficient (slope)
- b is the intercept

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to handle situations where the relationship between variables is clearly non-linear?

How can I apply this to a real project or problem?
I could use single variable linear regression to predict sales based on marketing spend, allowing for better budget allocation decisions.

What's a common misconception or edge case?
A common misconception is that correlation implies causation. Just because two variables show a strong linear relationship doesn't mean one causes the other - there might be confounding variables involved.

The key idea behind Linear Regression Single Variable is {{finding the best-fitting straight line that minimizes prediction error}}.

---
##### Tags

#ai/Linear_Regression_Single_Variable #ai #python #flashcard 