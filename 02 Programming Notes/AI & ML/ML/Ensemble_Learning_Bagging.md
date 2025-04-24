### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Ensemble Learning Bagging::Ensemble Learning Bagging (Bootstrap Aggregating) is a machine learning technique that improves model stability and accuracy by combining multiple models trained on different random subsets of the training data. It creates these subsets through bootstrap sampling—drawing observations with replacement from the original dataset. Each model in the ensemble makes independent predictions, which are then aggregated (typically by averaging for regression or majority voting for classification) to form a final prediction. This technique reduces variance, minimizes overfitting, and yields more robust predictions compared to individual models.

---
### Context  
Where and when is it used? Why is it important?

In what context is Ensemble Learning Bagging typically applied::Bagging is applied in situations requiring high-accuracy predictions with good generalization, such as in medical diagnosis, financial risk assessment, computer vision, and anomaly detection. It's particularly valuable when base models are sensitive to small data changes (like decision trees), when training data is noisy, or when model stability is crucial. Bagging is important because it provides significant accuracy improvements over single models, reduces overfitting, handles noisy data effectively, maintains model interpretability through feature importance averaging, enables parallel processing, and supplies built-in estimates of model uncertainty through the variance of ensemble predictions.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Decision_Tree]]
- [[Random_Forest]]
- [[Training_and_Testing_Data]]
- [[K_Fold_Cross_Validation]]
- [[Bias_vs_Variance]]
- [[Hyper_parameter_Tuning]]

What concepts are connected to Ensemble Learning Bagging::[[What_is_Machine_Learning]], [[Decision_Tree]], [[Random_Forest]], [[Training_and_Testing_Data]], [[K_Fold_Cross_Validation]], [[Bias_vs_Variance]], [[Hyper_parameter_Tuning]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Ensemble Learning - Bagging Implementation and Comparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import seaborn as sns
from collections import Counter

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Basic Bagging with Decision Trees
print("=== Basic Bagging with Decision Trees ===")

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"Dataset: Breast Cancer")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {target_names}")
print(f"Class distribution: {Counter(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a single decision tree (base model)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Create a bagging ensemble of decision trees
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,  # Number of base models
    max_samples=0.8,   # Size of bootstrap samples (80% of training data)
    bootstrap=True,    # Use bootstrap sampling
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)

# Create a Random Forest (specialized version of bagging)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the models
print("\nModel Performances:")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"Bagging Ensemble Accuracy: {accuracy_score(y_test, y_pred_bagging):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Part 2: Visualizing Decision Boundaries
print("\n=== Visualizing Decision Boundaries ===")

# Generate a more visual dataset for boundary visualization
X_moons, y_moons = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

# Train models
tree_moons = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_moons.fit(X_train_moons, y_train_moons)

bagging_moons = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
    n_estimators=50,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)
bagging_moons.fit(X_train_moons, y_train_moons)

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    # Define mesh grid
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and scatter points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

# Plot decision boundaries
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(tree_moons, X_moons, y_moons, 'Decision Tree Boundary')

plt.subplot(1, 2, 2)
plot_decision_boundary(bagging_moons, X_moons, y_moons, 'Bagging Ensemble Boundary')

plt.tight_layout()
plt.show()

# Calculate and print accuracies
tree_moons_accuracy = accuracy_score(y_test_moons, tree_moons.predict(X_test_moons))
bagging_moons_accuracy = accuracy_score(y_test_moons, bagging_moons.predict(X_test_moons))
print(f"Moon Dataset - Decision Tree Accuracy: {tree_moons_accuracy:.4f}")
print(f"Moon Dataset - Bagging Ensemble Accuracy: {bagging_moons_accuracy:.4f}")

# Part 3: Bagging with different base estimators
print("\n=== Bagging with Different Base Estimators ===")

# Create bagging ensembles with different base estimators
base_estimators = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

bagging_models = {}
for name, estimator in base_estimators.items():
    bagging_models[name] = BaggingClassifier(
        base_estimator=estimator,
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging_models[name].fit(X_train, y_train)

# Evaluate each bagging model
for name, model in bagging_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Bagging with {name} - Accuracy: {accuracy:.4f}")

# Part 4: Effect of number of estimators
print("\n=== Effect of Number of Estimators ===")

n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
accuracies = []

for n_estimators in n_estimators_range:
    # Create and fit bagging model
    bagging_model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_estimators,
        max_samples=0.8,
        bootstrap=True,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = bagging_model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracies, 'o-', linewidth=2)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Estimators on Bagging Performance')
plt.grid(True)
plt.show()

# Part 5: Effect of bootstrap sample size
print("\n=== Effect of Bootstrap Sample Size ===")

sample_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
sample_accuracies = []

for sample_size in sample_sizes:
    # Create and fit bagging model
    bagging_model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        max_samples=sample_size,
        bootstrap=True,
        random_state=42
    )
    bagging_model.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = bagging_model.predict(X_test)
    sample_accuracies.append(accuracy_score(y_test, y_pred))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, sample_accuracies, 'o-', linewidth=2)
plt.xlabel('Bootstrap Sample Size (fraction of training set)')
plt.ylabel('Accuracy')
plt.title('Effect of Bootstrap Sample Size on Bagging Performance')
plt.grid(True)
plt.show()

# Part 6: Out-of-Bag (OOB) Score
print("\n=== Out-of-Bag Score ===")

# Create bagging model with OOB score
bagging_oob = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,  # Enable OOB scoring
    random_state=42
)
bagging_oob.fit(X_train, y_train)

# Get OOB score
oob_score = bagging_oob.oob_score_
test_accuracy = accuracy_score(y_test, bagging_oob.predict(X_test))

print(f"Out-of-Bag Score: {oob_score:.4f}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(oob_score - test_accuracy):.4f}")

# Part 7: Feature Importance in Bagging
print("\n=== Feature Importance in Bagging ===")

# Random Forest feature importance (built-in)
rf_importances = rf.feature_importances_

# Permutation importance for bagging (since it doesn't have built-in feature_importances_)
perm_importance = permutation_importance(bagging, X_test, y_test, n_repeats=10, random_state=42)
bagging_importances = perm_importance.importances_mean

# Sort and get top features
rf_indices = np.argsort(rf_importances)[::-1]
bagging_indices = np.argsort(bagging_importances)[::-1]

top_n = 10  # Display top 10 features

print("\nRandom Forest Top Feature Importance:")
for i in range(top_n):
    print(f"{i+1}. {feature_names[rf_indices[i]]}: {rf_importances[rf_indices[i]]:.4f}")

print("\nBagging Permutation Importance:")
for i in range(top_n):
    print(f"{i+1}. {feature_names[bagging_indices[i]]}: {bagging_importances[bagging_indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.barh(range(top_n), rf_importances[rf_indices[:top_n]][::-1])
plt.yticks(range(top_n), [feature_names[i] for i in rf_indices[:top_n]][::-1])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')

plt.subplot(1, 2, 2)
plt.barh(range(top_n), bagging_importances[bagging_indices[:top_n]][::-1])
plt.yticks(range(top_n), [feature_names[i] for i in bagging_indices[:top_n]][::-1])
plt.title('Bagging Permutation Importance')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# Part 8: Bagging vs Random Forest - Detailed Comparison
print("\n=== Bagging vs Random Forest - Detailed Comparison ===")

# Create a regular bagging model with decision trees
bagging_dt = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)

# Create a bagging model with decision trees and feature sampling (similar to Random Forest)
bagging_dt_with_features = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,
    max_features=int(np.sqrt(X.shape[1])),  # Similar to Random Forest
    bootstrap=True,
    bootstrap_features=True,  # Sample features
    random_state=42
)

# Create a Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.8,  # bootstrap sample size
    random_state=42
)

# Train all models
models = {
    'Bagging (Decision Trees)': bagging_dt,
    'Bagging (DT + Feature Sampling)': bagging_dt_with_features,
    'Random Forest': rf_model
}

# Use 5-fold cross-validation to compare models
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Part 9: Implementation of Bootstrap Sampling from Scratch
print("\n=== Bootstrap Sampling Implementation from Scratch ===")

def bootstrap_sample(X, y):
    """Create a bootstrap sample from the data."""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def simple_bagging_from_scratch(X_train, y_train, X_test, base_model_class, n_estimators=10, **base_model_params):
    """Implement a simple bagging ensemble from scratch."""
    models = []
    
    # Train multiple models on bootstrap samples
    for i in range(n_estimators):
        # Create bootstrap sample
        X_boot, y_boot = bootstrap_sample(X_train, y_train)
        
        # Train model on bootstrap sample
        model = base_model_class(**base_model_params)
        model.fit(X_boot, y_boot)
        models.append(model)
    
    # Make predictions
    predictions = np.array([model.predict(X_test) for model in models])
    
    # Majority voting for classification
    final_pred = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=predictions
    )
    
    return final_pred, models

# Apply scratch implementation with small number of estimators for demonstration
scratch_pred, scratch_models = simple_bagging_from_scratch(
    X_train, y_train, X_test,
    base_model_class=DecisionTreeClassifier,
    n_estimators=10,
    random_state=42
)

scratch_accuracy = accuracy_score(y_test, scratch_pred)
print(f"Scratch Bagging Implementation Accuracy (10 estimators): {scratch_accuracy:.4f}")

# Part 10: Variance Reduction Demonstration
print("\n=== Variance Reduction Demonstration ===")

# Create a dataset with high variance for base models
X_noise, y_noise = make_moons(n_samples=500, noise=0.4, random_state=42)
X_train_noise, X_test_noise, y_train_noise, y_test_noise = train_test_split(
    X_noise, y_noise, test_size=0.3, random_state=42
)

# Train multiple individual decision trees
n_trees = 50
individual_preds = []
individual_accuracies = []

for i in range(n_trees):
    # Train with slightly different random states to simulate different samples
    tree = DecisionTreeClassifier(max_depth=4, random_state=i)
    tree.fit(X_train_noise, y_train_noise)
    
    pred = tree.predict(X_test_noise)
    individual_preds.append(pred)
    individual_accuracies.append(accuracy_score(y_test_noise, pred))

# Calculate variance of predictions across trees for each test point
prediction_variance = np.var(individual_preds, axis=0)
avg_prediction_variance = np.mean(prediction_variance)

# Ensemble prediction (majority vote)
ensemble_pred = np.apply_along_axis(
    lambda x: np.argmax(np.bincount(x)), 
    axis=0, 
    arr=np.array(individual_preds)
)
ensemble_accuracy = accuracy_score(y_test_noise, ensemble_pred)

print(f"Individual Trees - Mean Accuracy: {np.mean(individual_accuracies):.4f}")
print(f"Individual Trees - Accuracy Std Dev: {np.std(individual_accuracies):.4f}")
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print(f"Average Prediction Variance across Test Points: {avg_prediction_variance:.4f}")

# Visualize prediction variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_test_noise[:, 0], X_test_noise[:, 1], c=prediction_variance, cmap='viridis')
plt.colorbar(scatter, label='Prediction Variance')
plt.title('Prediction Variance Across Individual Trees')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
colors = ['red' if pred != y_test_noise[i] else 'green' for i, pred in enumerate(ensemble_pred)]
plt.scatter(X_test_noise[:, 0], X_test_noise[:, 1], c=colors)
plt.title('Ensemble Predictions (Green = Correct, Red = Wrong)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Part 11: Summary
print("\n=== Bagging Summary ===")
print("\nKey Concepts:")
print("1. Bootstrap sampling: Creating multiple datasets by sampling with replacement")
print("2. Training independent models on each bootstrap sample")
print("3. Aggregating predictions (voting or averaging)")
print("4. Out-of-Bag (OOB) estimation for model validation")

print("\nAdvantages of Bagging:")
print("- Reduces variance and overfitting")
print("- Improves stability and accuracy")
print("- Provides built-in estimate of generalization error (OOB)")
print("- Works well with 'high-variance' base models (e.g., decision trees)")
print("- Parallelizable (each model can be trained independently)")

print("\nDifferences between Bagging and Random Forest:")
print("- Bagging uses the same features for all models")
print("- Random Forest adds feature randomness (random subset of features at each split)")
print("- Random Forest often has additional tree-specific optimizations")
print("- Random Forest better decorrelates the base models")

print("\nUse Cases:")
print("- When base models are unstable (high variance)")
print("- When dataset has noise")
print("- When computing resources allow parallel training")
print("- When interpretability is still needed (vs. more complex ensembles)")
```

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to optimally balance the tradeoff between the number of base models in a bagging ensemble and the computational cost, especially for very large datasets? Is there a principled way to determine when adding more models provides diminishing returns in accuracy improvement?

How can I apply this to a real project or problem?
I could use bagging to improve the reliability of customer churn prediction by creating an ensemble of models trained on different subsets of customer data. The variance in predictions across models would also indicate the confidence level for each customer's churn probability, allowing the business to prioritize retention efforts for customers with the highest and most confidently predicted churn risk.

What's a common misconception or edge case?
A common misconception is that bagging always improves performance for any type of model. In reality, bagging provides the most benefit for "high-variance" models like deep decision trees that are prone to overfitting. For "high-bias" models like linear regression, bagging might provide minimal benefit since these models are already stable. Also, people sometimes confuse the Random Forest algorithm with basic bagging, not realizing that Random Forest incorporates additional feature randomization that further reduces correlation between the base models.

The key idea behind Ensemble Learning Bagging is {{combining multiple models trained on bootstrap samples to reduce variance and improve prediction stability by leveraging the wisdom of crowds effect}}.

---
##### Tags

#ai/Ensemble_Learning_Bagging #ai #python #flashcard 