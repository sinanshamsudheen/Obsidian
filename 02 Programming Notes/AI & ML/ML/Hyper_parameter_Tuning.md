### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Hyper parameter Tuning::Hyperparameter tuning is the process of systematically searching for the optimal configuration of hyperparameters (parameters that aren't learned during model training) for a machine learning algorithm. It involves defining a search space of possible hyperparameter values, evaluating model performance for different combinations using techniques like cross-validation, and selecting the combination that yields the best performance. GridSearchCV is a specific implementation in scikit-learn that performs an exhaustive search over a predefined grid of hyperparameter values.

---
### Context  
Where and when is it used? Why is it important?

In what context is Hyper parameter Tuning typically applied::Hyperparameter tuning is applied during model development in machine learning projects across domains like image classification, natural language processing, recommendation systems, and financial forecasting. It's used after choosing a model architecture but before final training, and when baseline models show potential for improvement. This process is important because hyperparameters significantly affect model performance and generalization ability, help prevent overfitting and underfitting, enable fair comparison between different models, automate what would otherwise be tedious manual parameter adjustments, and ultimately lead to more robust and reliable models in production.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Training_and_Testing_Data]]
- [[K_Fold_Cross_Validation]]
- [[Bias_vs_Variance]]
- [[Support_Vector_Machine]]
- [[Random_Forest]]
- [[Decision_Tree]]

What concepts are connected to Hyper parameter Tuning::[[What_is_Machine_Learning]], [[Training_and_Testing_Data]], [[K_Fold_Cross_Validation]], [[Bias_vs_Variance]], [[Support_Vector_Machine]], [[Random_Forest]], [[Decision_Tree]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import classification_report, make_scorer
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Grid Search CV for Classification
print("=== GridSearchCV for Classification ===")

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear']
}

# Create grid search object
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Perform grid search
print("Fitting GridSearchCV...")
start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

# Print results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Time taken: {grid_search_time:.2f} seconds")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy with best model: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Part 2: Randomized Search CV for faster hyperparameter optimization
print("\n=== RandomizedSearchCV for Classification ===")

# Define a broader parameter distribution for randomized search
param_distributions = {
    'classifier__C': uniform(0.1, 100),
    'classifier__gamma': uniform(0.001, 1),
    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# Create randomized search object
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=20,  # Number of parameter settings to sample
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Perform randomized search
print("Fitting RandomizedSearchCV...")
start_time = time.time()
random_search.fit(X_train, y_train)
random_search_time = time.time() - start_time

# Print results
print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")
print(f"Time taken: {random_search_time:.2f} seconds")

# Compare efficiency
print(f"\nEfficiency comparison:")
print(f"GridSearchCV: {grid_search_time:.2f} seconds")
print(f"RandomizedSearchCV: {random_search_time:.2f} seconds")
print(f"Speed-up: {grid_search_time / random_search_time:.2f}x")
print(f"Grid Search best score: {grid_search.best_score_:.4f}")
print(f"Randomized Search best score: {random_search.best_score_:.4f}")

# Part 3: Hyperparameter Tuning for Regression
print("\n=== Hyperparameter Tuning for Regression ===")

# Load regression dataset (a subset of California housing)
housing = fetch_california_housing()
X_reg = housing.data[:2000]  # Using a subset for faster computation
y_reg = housing.target[:2000]
reg_feature_names = housing.feature_names

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create pipeline
reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor())
])

# Define parameter grid
reg_param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5, 10]
}

# Create grid search
reg_grid_search = GridSearchCV(
    reg_pipeline,
    reg_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
print("Fitting GridSearchCV for regression...")
reg_grid_search.fit(X_train_reg, y_train_reg)

# Print results
print(f"Best parameters: {reg_grid_search.best_params_}")
print(f"Best cross-validation MSE: {-reg_grid_search.best_score_:.4f}")

# Evaluate on test set
reg_best_model = reg_grid_search.best_estimator_
y_pred_reg = reg_best_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Test set MSE: {mse:.4f}")
print(f"Test set R²: {r2:.4f}")

# Part 4: Custom Scoring Metrics
print("\n=== Custom Scoring Metrics for Hyperparameter Tuning ===")

# Define a custom scoring function that prioritizes recall for the minority class (malignant)
def custom_recall_score(y_true, y_pred):
    # Focus on recall for the positive class (malignant=1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

# Create a scorer from the custom function
custom_scorer = make_scorer(custom_recall_score)

# Create a new grid search with the custom scorer
custom_grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring=custom_scorer,  # Use custom scorer
    n_jobs=-1,
    verbose=1
)

# Fit grid search with custom scorer
print("Fitting GridSearchCV with custom scorer...")
custom_grid_search.fit(X_train, y_train)

# Print results
print(f"Best parameters (custom scorer): {custom_grid_search.best_params_}")
print(f"Best cross-validation score (custom scorer): {custom_grid_search.best_score_:.4f}")

# Evaluate with different metrics
best_model_custom = custom_grid_search.best_estimator_
y_pred_custom = best_model_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Test set accuracy: {accuracy_custom:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, target_names=target_names))

# Compare with standard scoring
print("\nComparison of models optimized with different scoring metrics:")
print(f"Standard accuracy-optimized model parameters: {grid_search.best_params_}")
print(f"Custom recall-optimized model parameters: {custom_grid_search.best_params_}")

# Part 5: Visualizing Hyperparameter Effects
print("\n=== Visualizing Hyperparameter Effects ===")

# Validation curve for SVM's C parameter
C_range = np.logspace(-2, 3, 6)
train_scores, test_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale'), 
    X, y,
    param_name="C", 
    param_range=C_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Calculate mean and std for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.title("Validation Curve for SVM (C parameter)")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.semilogx(C_range, train_mean, label="Training score", color="blue")
plt.fill_between(C_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.semilogx(C_range, test_mean, label="Cross-validation score", color="red")
plt.fill_between(C_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Part 6: Advanced Technique - Hyperparameter Tuning for Multiple Models
print("\n=== Comparing Multiple Models with Hyperparameter Tuning ===")

# Define models with their parameter grids
models = {
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    }
}

# Function to find best model among multiple options
def find_best_model(X, y, models, cv=5):
    # Scale the data once (since we're not using Pipeline here for simplicity)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_score = -np.inf
    best_model = None
    best_params = None
    model_results = {}
    
    for name, model_info in models.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_scaled, y)
        
        model_results[name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_
        }
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = name
            best_params = grid_search.best_params_
    
    return model_results, best_model_name, best_model, best_params, best_score

# Find the best model
model_results, best_model_name, best_model, best_params, best_score = find_best_model(X, y, models)

# Print results
print("\nModel Comparison Results:")
for name, result in model_results.items():
    print(f"{name}:")
    print(f"  Best Score: {result['best_score']:.4f}")
    print(f"  Best Parameters: {result['best_params']}")

print(f"\nOverall Best Model: {best_model_name}")
print(f"Best Score: {best_score:.4f}")
print(f"Best Parameters: {best_params}")

# Part 7: Learning Curves to Identify Overfitting/Underfitting During Tuning
print("\n=== Learning Curves for Model Diagnosis ===")

# Get the best SVM model from previous grid search
best_svm = grid_search.best_estimator_

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    best_svm, 
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Calculate mean and std for training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.title("Learning Curve for Best SVM Model")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.plot(train_sizes, test_mean, 'o-', color="red", label="Cross-validation score")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
plt.legend(loc="best")
plt.show()

# Part 8: Nested Cross-Validation for Unbiased Evaluation
print("\n=== Nested Cross-Validation ===")

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Define the model with a parameter grid
model = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# Create an inner and outer cross-validation loop
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Create the nested CV model
nested_grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=inner_cv, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# Perform nested cross-validation
nested_scores = cross_val_score(nested_grid_search, X, y, cv=outer_cv, n_jobs=-1)

# Print results
print(f"Nested CV Scores: {nested_scores}")
print(f"Mean Nested CV Score: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
print(f"Regular CV Score (from earlier): {grid_search.best_score_:.4f}")

# Part 9: Visualizations and Result Analysis
print("\n=== Result Visualization ===")

# 1. Heatmap of parameter combinations for SVM (C vs gamma)
plt.figure(figsize=(12, 10))

# Extract results from grid search
results = pd.DataFrame(grid_search.cv_results_)

# Filter results for rbf kernel
rbf_results = results[results['param_classifier__kernel'] == 'rbf']

# Create a pivot table of C vs gamma
C_values = sorted(rbf_results['param_classifier__C'].unique())
gamma_values = [g for g in rbf_results['param_classifier__gamma'].unique() if isinstance(g, (int, float))]
gamma_values = sorted(gamma_values)

# Create score matrix
score_matrix = np.zeros((len(C_values), len(gamma_values)))
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        mask = (rbf_results['param_classifier__C'] == C) & (rbf_results['param_classifier__gamma'] == gamma)
        if mask.any():
            score_matrix[i, j] = rbf_results.loc[mask, 'mean_test_score'].values[0]

# Plot heatmap
plt.subplot(1, 1, 1)
sns.heatmap(score_matrix, annot=True, fmt='.4f', cmap='viridis',
           xticklabels=gamma_values, yticklabels=C_values)
plt.xlabel('gamma')
plt.ylabel('C')
plt.title('Grid Search Scores for SVM with RBF Kernel')
plt.tight_layout()
plt.show()

# 2. Parallel coordinates plot to visualize parameter relationships
def create_param_combinations(cv_results):
    params = []
    for key in cv_results['params'][0].keys():
        if key.startswith('classifier__'):
            params.append(key.replace('classifier__', ''))
    return params

# Get parameter names
param_names = create_param_combinations(grid_search.cv_results_)

# Convert results to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Extract relevant parameters and scores
param_cols = ['param_' + col for col in ['classifier__' + param for param in param_names]]
results_df_filtered = results_df[param_cols + ['mean_test_score']]

# Rename columns for clarity
rename_dict = {param_col: param_col.replace('param_classifier__', '') for param_col in param_cols}
rename_dict['mean_test_score'] = 'Score'
results_df_filtered = results_df_filtered.rename(columns=rename_dict)

# Convert categorical variables to numeric for plotting
for col in results_df_filtered.columns:
    if col != 'Score' and results_df_filtered[col].dtype == 'object':
        unique_vals = results_df_filtered[col].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        results_df_filtered[col + '_numeric'] = results_df_filtered[col].map(mapping)
        # Store mapping for reference
        print(f"Mapping for {col}: {mapping}")

# Normalize data for parallel coordinates plot
from sklearn.preprocessing import MinMaxScaler
numeric_cols = [col for col in results_df_filtered.columns if col.endswith('_numeric') or col == 'Score']
scaler = MinMaxScaler()
results_df_filtered[numeric_cols] = scaler.fit_transform(results_df_filtered[numeric_cols])

# Create parallel coordinates plot
plt.figure(figsize=(15, 8))
pd.plotting.parallel_coordinates(
    results_df_filtered, 
    'Score', 
    cols=[col for col in numeric_cols if col != 'Score'],
    colormap='viridis'
)
plt.title('Parallel Coordinates Plot of Hyperparameter Combinations')
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary of key findings
print("\n=== Summary of Hyperparameter Tuning ===")
print(f"1. Best model: {best_model_name} with parameters {best_params}")
print(f"2. Cross-validation accuracy: {best_score:.4f}")
print(f"3. Test set accuracy: {accuracy:.4f}")
print("4. Key hyperparameter insights:")
print("   - For SVM: Balance between C (regularization) and gamma (kernel coefficient) is critical")
print("   - For Random Forest: Number of trees and max depth are main performance drivers")
print("   - For all models: Proper preprocessing and feature scaling are essential")
print("5. Recommendations:")
print("   - Start with RandomizedSearchCV for efficient exploration")
print("   - Refine promising areas with GridSearchCV")
print("   - Always use cross-validation to prevent overfitting to validation data")
print("   - Consider the computational cost vs. performance improvement tradeoff")
```

**Hyperparameter Tuning Approaches**:

1. **Grid Search (GridSearchCV)**:
   - Exhaustive search over a predefined parameter grid
   - Systematically tries all combinations
   - Guaranteed to find the best combination within the grid
   - Computationally expensive for large parameter spaces

2. **Random Search (RandomizedSearchCV)**:
   - Samples random combinations from parameter distributions
   - More efficient for high-dimensional spaces
   - Can find good solutions with fewer iterations
   - May miss the absolute best combination

3. **Bayesian Optimization**:
   - Builds a probabilistic model of the objective function
   - Uses past evaluations to guide future sampling
   - Particularly effective for expensive-to-evaluate models
   - Examples: scikit-optimize, Hyperopt, Optuna

4. **Evolutionary Algorithms**:
   - Uses concepts like mutation, crossover, and selection
   - Can efficiently navigate complex parameter spaces
   - Useful for very high-dimensional problems

**Tips for Efficient Hyperparameter Tuning**:

1. Start with a broad, coarse search using RandomizedSearchCV
2. Refine promising regions with a focused GridSearchCV
3. Use logarithmic scales for numeric parameters with wide ranges
4. Leverage domain knowledge to set reasonable parameter boundaries
5. Monitor for overfitting during tuning with validation curves
6. Consider computational resources and use parallelization when possible

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to efficiently tune hyperparameters for computationally expensive models or very large datasets, where even a single model training might take hours or days?

How can I apply this to a real project or problem?
I could implement a loan default prediction system for a financial institution, using GridSearchCV to optimize a Random Forest model's hyperparameters (tree depth, number of estimators, minimum samples per leaf) to find the best balance between maximizing prediction accuracy and minimizing false negatives, which would be particularly costly in this application.

What's a common misconception or edge case?
A common misconception is that more extensive hyperparameter tuning always leads to better models. In reality, excessively tuning hyperparameters can lead to overfitting on the validation data, making the model perform worse on new, unseen data. This "overfitting to the validation set" often happens when the validation strategy doesn't properly represent the distribution of future data or when the model is evaluated too many times on the same validation set.

The key idea behind Hyper parameter Tuning is {{systematically exploring the hyperparameter space to find the optimal configuration that maximizes model performance on validation data while maintaining generalization ability}}.

---
##### Tags

#ai/Hyper_parameter_Tuning #ai #python #flashcard 