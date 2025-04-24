### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of L1 and L2 Regularization::L1 and L2 Regularization are techniques to prevent overfitting in machine learning models by adding penalty terms to the loss function based on the magnitude of model coefficients. L1 regularization (Lasso) adds a penalty equal to the absolute value of coefficients, which can drive some coefficients to exactly zero, effectively performing feature selection. L2 regularization (Ridge) adds a penalty equal to the square of coefficients, which shrinks coefficients toward zero but rarely makes them exactly zero, helping to handle multicollinearity.

---
### Context  
Where and when is it used? Why is it important?

In what context is L1 and L2 Regularization typically applied::L1 and L2 regularization are applied in supervised learning models, particularly in linear and logistic regression, neural networks, and SVMs when there's a risk of overfitting due to high model complexity, high-dimensional data, or limited training samples. They're important because they improve model generalization by controlling complexity, prevent overfitting by constraining coefficient values, can handle multicollinearity (L2) or perform automatic feature selection (L1), and provide a mathematical framework to balance fitting the data versus model simplicity.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Linear_Regression_Multiple_Variables]]
- [[Logistic_Regression_Binary_Classification]]
- [[Logistic_Regression_Multiclass_Classification]]
- [[Bias_vs_Variance]]
- [[Hyper_parameter_Tuning]]
- [[Support_Vector_Machine]]

What concepts are connected to L1 and L2 Regularization::[[What_is_Machine_Learning]], [[Linear_Regression_Multiple_Variables]], [[Logistic_Regression_Binary_Classification]], [[Bias_vs_Variance]], [[Hyper_parameter_Tuning]], [[Support_Vector_Machine]], [[Gradient_Descent_and_Cost_Function]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# L1 and L2 Regularization Implementation and Comparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset with some irrelevant features
def generate_dataset(n_samples=200, n_features=50, n_informative=10, noise=10.0):
    """Generate a regression dataset with many features but only some are informative."""
    X, y, coef = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative,
        noise=noise, 
        coef=True,  # Return the ground truth coefficients
        random_state=42
    )
    
    # Scale features for better visualization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create a DataFrame for easier handling
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Create a true coefficients array with zeros for non-informative features
    true_coef = np.zeros(n_features)
    true_coef[:n_informative] = coef
    
    return df, y, true_coef, feature_names

# Generate data
X_df, y, true_coef, feature_names = generate_dataset()
X = X_df.values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Part 1: Compare standard linear regression with Ridge (L2) and Lasso (L1)
print("=== Comparing Standard, Ridge, and Lasso Regression ===")

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'ElasticNet (L1+L2)': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Train and evaluate each model
results = {}
coefficients = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Store results
    results[name] = {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'R² Score': r2
    }
    
    # Store coefficients (for later visualization)
    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_
    
    print(f"{name}:")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Count non-zero coefficients (to see feature selection effect)
    if hasattr(model, 'coef_'):
        n_nonzero = np.sum(model.coef_ != 0)
        print(f"  Non-zero coefficients: {n_nonzero}/{len(model.coef_)}")
    print()

# Part 2: Visual comparison of coefficients
plt.figure(figsize=(14, 8))

# Plot ground truth coefficients
plt.subplot(3, 2, 1)
plt.stem(range(len(true_coef)), true_coef)
plt.title('Ground Truth Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.grid(True)

# Plot coefficients for each model
for i, (name, coef) in enumerate(coefficients.items(), 2):
    plt.subplot(3, 2, i)
    plt.stem(range(len(coef)), coef)
    plt.title(f'{name} Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Part 3: Effect of regularization strength (alpha)
print("\n=== Effect of Regularization Strength (alpha) ===")

# Range of alpha values to test
alphas = np.logspace(-4, 2, 10)  # From 0.0001 to 100

ridge_train_scores = []
ridge_test_scores = []
lasso_train_scores = []
lasso_test_scores = []
ridge_n_nonzero = []
lasso_n_nonzero = []

for alpha in alphas:
    # Ridge (L2)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_train_scores.append(mean_squared_error(y_train, ridge.predict(X_train)))
    ridge_test_scores.append(mean_squared_error(y_test, ridge.predict(X_test)))
    ridge_n_nonzero.append(np.sum(ridge.coef_ != 0))
    
    # Lasso (L1)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_train_scores.append(mean_squared_error(y_train, lasso.predict(X_train)))
    lasso_test_scores.append(mean_squared_error(y_test, lasso.predict(X_test)))
    lasso_n_nonzero.append(np.sum(lasso.coef_ != 0))

# Plot results
plt.figure(figsize=(14, 10))

# Plot 1: Ridge MSE vs Alpha
plt.subplot(2, 2, 1)
plt.semilogx(alphas, ridge_train_scores, 'b-o', label='Train MSE')
plt.semilogx(alphas, ridge_test_scores, 'r-o', label='Test MSE')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: MSE vs Alpha')
plt.legend()
plt.grid(True)

# Plot 2: Lasso MSE vs Alpha
plt.subplot(2, 2, 2)
plt.semilogx(alphas, lasso_train_scores, 'b-o', label='Train MSE')
plt.semilogx(alphas, lasso_test_scores, 'r-o', label='Test MSE')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression: MSE vs Alpha')
plt.legend()
plt.grid(True)

# Plot 3: Ridge - Number of Non-zero Coefficients vs Alpha
plt.subplot(2, 2, 3)
plt.semilogx(alphas, ridge_n_nonzero, 'g-o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Number of Non-zero Coefficients')
plt.title('Ridge: Number of Features Used vs Alpha')
plt.grid(True)

# Plot 4: Lasso - Number of Non-zero Coefficients vs Alpha
plt.subplot(2, 2, 4)
plt.semilogx(alphas, lasso_n_nonzero, 'g-o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Number of Non-zero Coefficients')
plt.title('Lasso: Number of Features Used vs Alpha')
plt.grid(True)

plt.tight_layout()
plt.show()

# Part 4: L1 vs L2 Regularization with GridSearchCV
print("\n=== Hyperparameter Tuning with GridSearchCV ===")

# Define parameter grids
ridge_param_grid = {'alpha': np.logspace(-4, 2, 20)}
lasso_param_grid = {'alpha': np.logspace(-4, 2, 20)}

# Create grid searches
ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit grid searches
ridge_grid.fit(X_train, y_train)
lasso_grid.fit(X_train, y_train)

# Print results
print(f"Ridge best alpha: {ridge_grid.best_params_['alpha']:.6f}")
print(f"Lasso best alpha: {lasso_grid.best_params_['alpha']:.6f}")

# Evaluate best models
ridge_best = ridge_grid.best_estimator_
lasso_best = lasso_grid.best_estimator_

ridge_test_mse = mean_squared_error(y_test, ridge_best.predict(X_test))
lasso_test_mse = mean_squared_error(y_test, lasso_best.predict(X_test))

print(f"Ridge best model test MSE: {ridge_test_mse:.4f}")
print(f"Lasso best model test MSE: {lasso_test_mse:.4f}")

# Count non-zero coefficients in best models
ridge_nonzero = np.sum(ridge_best.coef_ != 0)
lasso_nonzero = np.sum(lasso_best.coef_ != 0)
print(f"Ridge non-zero coefficients: {ridge_nonzero}/{len(ridge_best.coef_)}")
print(f"Lasso non-zero coefficients: {lasso_nonzero}/{len(lasso_best.coef_)}")

# Part 5: Learning Curves to Visualize Overfitting and Regularization Effect
print("\n=== Learning Curves: Impact of Regularization on Model Generalization ===")

# Function to plot learning curves
def plot_learning_curves(estimator, title, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    # Convert MSE to positive values
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.show()

# Plot learning curves for different models
plot_learning_curves(LinearRegression(), "Learning Curve: Standard Linear Regression", X, y)
plot_learning_curves(ridge_best, f"Learning Curve: Ridge (alpha={ridge_grid.best_params_['alpha']:.6f})", X, y)
plot_learning_curves(lasso_best, f"Learning Curve: Lasso (alpha={lasso_grid.best_params_['alpha']:.6f})", X, y)

# Part 6: Coefficient Paths (how coefficients change with alpha)
print("\n=== Coefficient Paths for L1 Regularization ===")

# Compute coefficient paths for Lasso
alphas_path = np.logspace(-1, 1, 50)
coefs = []

for alpha in alphas_path:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

# Only plot a subset of coefficients for clarity
plt.figure(figsize=(12, 8))
for i in range(15):  # Plot first 15 coefficients
    plt.plot(alphas_path, [coef[i] for coef in coefs], label=f'Feature {i+1}')

plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Part 7: Mathematical Representation and Comparison
print("\n=== Mathematical Representation of L1 and L2 Regularization ===")
print("Linear Regression Cost Function: J(θ) = MSE(θ)")
print("\nL1 Regularization (Lasso): J(θ) = MSE(θ) + α * Σ|θᵢ|")
print("- Adds absolute value of coefficients")
print("- Tends to produce sparse models (feature selection)")
print("- Not differentiable at θᵢ = 0")
print("\nL2 Regularization (Ridge): J(θ) = MSE(θ) + α * Σθᵢ²")
print("- Adds squared value of coefficients")
print("- Tends to spread out coefficient values")
print("- Keeps all features but shrinks their impact")
print("- Differentiable everywhere")
print("\nElasticNet: J(θ) = MSE(θ) + α * (ρ * Σ|θᵢ| + (1-ρ) * Σθᵢ²)")
print("- Combines L1 and L2 penalties")
print("- ρ controls the balance between L1 and L2")
print("- Useful when dealing with correlated features")

# Part 8: Summary of Key Characteristics
print("\n=== Summary of Key Characteristics ===")
print("L1 Regularization (Lasso):")
print("  - Sparsity: Can reduce coefficients to exactly zero")
print("  - Feature Selection: Automatically selects important features")
print("  - Simplicity: Produces simpler models with fewer parameters")
print("  - Handling High Dimensionality: Good for high-dimensional data")
print("  - Best For: When you suspect many features are irrelevant")
print("\nL2 Regularization (Ridge):")
print("  - Stability: Handles multicollinearity well")
print("  - Smoothness: Shrinks coefficients but rarely to exactly zero")
print("  - Numerical Stability: Has a closed-form solution")
print("  - Regularization Effect: More uniform regularization of coefficients")
print("  - Best For: When most features contribute to the outcome")
print("\nElasticNet:")
print("  - Flexibility: Combines advantages of both L1 and L2")
print("  - Control: Allows fine-tuning the regularization type")
print("  - Robustness: Works well with correlated features")
print("  - Best For: When you want some sparsity but not fully sparse solutions")
```

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to efficiently determine the optimal regularization strength (alpha) and balance between L1 and L2 penalties in ElasticNet for different types of datasets without extensive grid searches?

How can I apply this to a real project or problem?
I could apply L1 regularization to a marketing analytics model predicting customer lifetime value, where we have hundreds of potential customer attributes but want to identify the most influential features for targeted marketing campaigns, making the model both more interpretable and computationally efficient.

What's a common misconception or edge case?
A common misconception is that regularization is only useful for preventing overfitting in large models. In reality, even relatively simple models can benefit from regularization when data is noisy or contains multicollinearity. Another misconception is that L1 regularization always performs better feature selection than explicit feature selection methods, but in practice, domain knowledge combined with statistical feature selection may outperform blind L1 regularization.

The key idea behind L1 and L2 Regularization is {{controlling model complexity by adding penalty terms to the loss function, with L1 promoting sparsity and feature selection while L2 handles multicollinearity through coefficient shrinkage}}.

---
##### Tags

#ai/L1_and_L2_Regularization #ai #python #flashcard 