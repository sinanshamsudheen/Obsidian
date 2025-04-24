### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Gradient Descent and Cost Function::Gradient Descent is an optimization algorithm used to minimize a cost function by iteratively adjusting parameters in the direction of steepest descent of the gradient. The Cost Function (or Loss Function) quantifies how well a model's predictions match the actual values, with lower values indicating better performance. Together, they form the mathematical foundation for training many machine learning models.

---
### Context  
Where and when is it used? Why is it important?

In what context is Gradient Descent and Cost Function typically applied::Gradient Descent and Cost Functions are applied when training parametric machine learning models, particularly in neural networks, linear regression, and logistic regression. They're used to find optimal model parameters when no closed-form solution exists or when the dataset is too large for direct computation. These concepts are important because they enable efficient model training, provide a quantitative way to measure model performance, and allow us to balance between underfitting and overfitting.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Linear_Regression_Single_Variable]]
- [[Linear_Regression_Multiple_Variables]]
- [[Logistic_Regression_Binary_Classification]]
- [[L1_and_L2_Regularization]]
- [[Bias_vs_Variance]]

What concepts are connected to Gradient Descent and Cost Function::[[What_is_Machine_Learning]], [[Linear_Regression_Single_Variable]], [[Linear_Regression_Multiple_Variables]], [[Logistic_Regression_Binary_Classification]], [[L1_and_L2_Regularization]], [[Bias_vs_Variance]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Implementing Gradient Descent for Linear Regression from scratch
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (intercept)
X_b = np.c_[np.ones((100, 1)), X]

# Initialize parameters
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

# Lists to store cost history
cost_history = []

# Gradient Descent
for iteration in range(n_iterations):
    # Compute predictions
    predictions = X_b.dot(theta)
    
    # Compute errors
    errors = predictions - y
    
    # Compute gradients
    gradients = 2/m * X_b.T.dot(errors)
    
    # Update parameters
    theta = theta - learning_rate * gradients
    
    # Compute cost (Mean Squared Error)
    cost = np.mean(errors ** 2)
    cost_history.append(cost)
    
    # Print progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Cost: {cost:.4f}")

print(f"Final parameters: Intercept = {theta[0][0]:.4f}, Slope = {theta[1][0]:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Minimization with Gradient Descent')
plt.grid(True)
plt.show()

# Plot the data and the final regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.grid(True)
plt.show()
```

Common cost functions:
- Mean Squared Error (MSE): (1/n) * Σ(yᵢ - ŷᵢ)²
- Binary Cross-Entropy: -Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
- Categorical Cross-Entropy: -Σ Σ yᵢⱼlog(ŷᵢⱼ)

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to effectively choose between different variants of gradient descent (batch, mini-batch, stochastic) for specific problems?

How can I apply this to a real project or problem?
I could use custom loss functions to handle imbalanced datasets or to prioritize certain types of prediction errors over others based on business requirements.

What's a common misconception or edge case?
A common misconception is that gradient descent always converges to the global minimum. For non-convex cost functions (like those in neural networks), it may get stuck in local minima, which is why techniques like momentum and learning rate scheduling are important.

The key idea behind Gradient Descent and Cost Function is {{iteratively adjusting model parameters to minimize prediction error}}.

---
##### Tags

#ai/Gradient_Descent_and_Cost_Function #ai #python #flashcard 