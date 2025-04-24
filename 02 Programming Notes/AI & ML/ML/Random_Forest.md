### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Random Forest::Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (for classification) or mean prediction (for regression) of the individual trees. It introduces randomness by building each tree on a random subset of the data (bootstrap sampling) and considering only a random subset of features at each split, which helps reduce overfitting and improve generalization compared to single decision trees.

---
### Context  
Where and when is it used? Why is it important?

In what context is Random Forest typically applied::Random Forest is applied in diverse domains for both classification and regression tasks including finance (fraud detection, credit scoring), healthcare (disease prediction), remote sensing (land cover classification), marketing (customer segmentation), and recommendation systems. It's important because it provides high prediction accuracy without requiring extensive hyperparameter tuning, handles high-dimensional data well, can identify important features, is less prone to overfitting than single decision trees, maintains good performance with missing or unbalanced data, and can be used for unsupervised learning (via proximity measures).

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Decision_Tree]]
- [[Ensemble_Learning_Bagging]]
- [[Training_and_Testing_Data]]
- [[K_Fold_Cross_Validation]]
- [[Hyper_parameter_Tuning]]
- [[Bias_vs_Variance]]

What concepts are connected to Random Forest::[[What_is_Machine_Learning]], [[Decision_Tree]], [[Ensemble_Learning_Bagging]], [[Training_and_Testing_Data]], [[K_Fold_Cross_Validation]], [[Hyper_parameter_Tuning]], [[Bias_vs_Variance]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Random Forest example for both classification and regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Part 1: Random Forest for Classification
print("=== Random Forest Classification ===")

# Load the breast cancer dataset
cancer = load_breast_cancer()
X_class = cancer.data
y_class = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

print(f"Dataset shape: {X_class.shape}")
print(f"Number of features: {X_class.shape[1]}")
print(f"Classes: {target_names}")
print(f"Class distribution: {np.bincount(y_class)}")

# Split data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,    # Maximum depth of trees (None means unlimited)
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,    # Use bootstrap sampling
    random_state=42
)

rf_classifier.fit(X_train_class, y_train_class)

# Make predictions
y_pred_class = rf_classifier.predict(X_test_class)
y_pred_proba = rf_classifier.predict_proba(X_test_class)[:, 1]  # Probability for positive class

# Evaluate the classifier
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"\nRandom Forest Classifier Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class, target_names=target_names))

# Calculate feature importance
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature ranking (top 10):")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

# Part 2: Random Forest for Regression
print("\n=== Random Forest Regression ===")

# Load a subset of the California housing dataset
housing = fetch_california_housing()
X_reg = housing.data[:2000]  # Using a subset for faster execution
y_reg = housing.target[:2000]
reg_feature_names = housing.feature_names

print(f"Regression dataset shape: {X_reg.shape}")
print(f"Features: {reg_feature_names}")

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train a Random Forest regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_regressor.predict(X_test_reg)

# Evaluate the regressor
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nRandom Forest Regressor Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Calculate feature importance for regression
reg_importances = rf_regressor.feature_importances_
reg_indices = np.argsort(reg_importances)[::-1]

print("\nRegression feature ranking:")
for i in range(len(reg_feature_names)):
    print(f"{i+1}. {reg_feature_names[reg_indices[i]]} ({reg_importances[reg_indices[i]]:.4f})")

# Part 3: Cross-validation
print("\n=== Cross-validation ===")
cv_scores_class = cross_val_score(rf_classifier, X_class, y_class, cv=5, scoring='accuracy')
cv_scores_reg = cross_val_score(rf_regressor, X_reg, y_reg, cv=5, scoring='r2')

print(f"Classification 5-fold CV accuracy: {cv_scores_class.mean():.4f} ± {cv_scores_class.std():.4f}")
print(f"Regression 5-fold CV R²: {cv_scores_reg.mean():.4f} ± {cv_scores_reg.std():.4f}")

# Part 4: Effect of number of trees (n_estimators)
print("\n=== Effect of number of trees ===")
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train_class, y_train_class)
    train_scores.append(accuracy_score(y_train_class, rf.predict(X_train_class)))
    test_scores.append(accuracy_score(y_test_class, rf.predict(X_test_class)))

# Part 5: Visualizations
plt.figure(figsize=(20, 12))

# Plot 1: Feature importances for classification
plt.subplot(2, 2, 1)
plt.barh(range(10), importances[indices[:10]], align='center')
plt.yticks(range(10), [feature_names[i] for i in indices[:10]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features (Classification)')

# Plot 2: Feature importances for regression
plt.subplot(2, 2, 2)
plt.barh(range(len(reg_feature_names)), reg_importances[reg_indices], align='center')
plt.yticks(range(len(reg_feature_names)), [reg_feature_names[i] for i in reg_indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance (Regression)')

# Plot 3: Effect of number of trees on performance
plt.subplot(2, 2, 3)
plt.plot(n_estimators_range, train_scores, 'o-', label='Training accuracy')
plt.plot(n_estimators_range, test_scores, 'o-', label='Testing accuracy')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Effect of Forest Size on Performance')
plt.legend()
plt.grid(True)
plt.xscale('log')

# Plot 4: Confusion matrix
plt.subplot(2, 2, 4)
cm = confusion_matrix(y_test_class, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Classification)')

plt.tight_layout()
plt.show()

# Part 6: Permutation Feature Importance (more robust than default feature_importances_)
print("\n=== Permutation Feature Importance ===")
perm_importance = permutation_importance(rf_classifier, X_test_class, y_test_class, 
                                        n_repeats=10, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]
plt.figure(figsize=(12, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T, 
           vert=False, labels=[feature_names[i] for i in sorted_idx[:10]])
plt.title("Permutation Importances (Top 10 features)")
plt.tight_layout()
plt.show()

# Part 7: Out-of-Bag (OOB) error estimation
print("\n=== Out-of-Bag Error Estimation ===")
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train_class, y_train_class)
print(f"OOB accuracy score: {rf_oob.oob_score_:.4f}")

# Compare OOB score with test set
test_accuracy = accuracy_score(y_test_class, rf_oob.predict(X_test_class))
print(f"Test set accuracy: {test_accuracy:.4f}")
print(f"Difference: {abs(rf_oob.oob_score_ - test_accuracy):.4f}")

# Part 8: Demonstrate how Random Forest reduces overfitting compared to a single Decision Tree
from sklearn.tree import DecisionTreeClassifier

print("\n=== Random Forest vs Decision Tree (Overfitting) ===")
# Create a deep decision tree that will overfit
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_class, y_train_class)

dt_train_accuracy = accuracy_score(y_train_class, dt.predict(X_train_class))
dt_test_accuracy = accuracy_score(y_test_class, dt.predict(X_test_class))

rf_train_accuracy = accuracy_score(y_train_class, rf_classifier.predict(X_train_class))
rf_test_accuracy = accuracy_score(y_test_class, rf_classifier.predict(X_test_class))

print(f"Decision Tree - Training accuracy: {dt_train_accuracy:.4f}, Test accuracy: {dt_test_accuracy:.4f}")
print(f"Random Forest - Training accuracy: {rf_train_accuracy:.4f}, Test accuracy: {rf_test_accuracy:.4f}")
print(f"Overfitting gap (train-test) - Decision Tree: {dt_train_accuracy - dt_test_accuracy:.4f}")
print(f"Overfitting gap (train-test) - Random Forest: {rf_train_accuracy - rf_test_accuracy:.4f}")

# Visual comparison of overfitting
plt.figure(figsize=(10, 6))
models = ['Decision Tree', 'Random Forest']
train_accuracies = [dt_train_accuracy, rf_train_accuracy]
test_accuracies = [dt_test_accuracy, rf_test_accuracy]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
plt.bar(x + width/2, test_accuracies, width, label='Testing Accuracy')

plt.ylabel('Accuracy')
plt.title('Decision Tree vs Random Forest: Overfitting Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True, axis='y')
plt.show()
```

**Key principles behind Random Forest:**

1. **Bootstrap Aggregating (Bagging)**: Each tree is trained on a random subset of the data sampled with replacement (bootstrap sample), typically around 63.2% of the original data for each tree.

2. **Feature Randomness**: At each node split, only a random subset of features is considered (typically √p features for classification and p/3 for regression, where p is the total number of features).

3. **Ensemble Decision Making**:
   - Classification: Majority vote among all trees
   - Regression: Average of all tree predictions

4. **Out-of-Bag (OOB) Estimation**: The samples not used for training a particular tree (about 36.8% of data) are used to estimate its performance, providing an internal validation mechanism.

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to optimally set the hyperparameters for the randomness in Random Forest (mtry - the number of variables randomly sampled at each split) for different types of datasets?

How can I apply this to a real project or problem?
I could use Random Forest to predict customer churn by analyzing historical customer data, identifying the most influential factors, and then using these insights to implement targeted retention strategies for at-risk customers.

What's a common misconception or edge case?
A common misconception is that Random Forests always need a large number of trees to perform well. In practice, the performance improvement typically plateaus after a certain number of trees (often around 100-200), and adding more trees mainly increases computational cost without significant accuracy gains. The optimal number depends on the specific dataset and problem.

The key idea behind Random Forest is {{combining multiple randomized decision trees to create a more robust, accurate model that mitigates the overfitting tendencies of individual trees}}.

---
##### Tags

#ai/Random_Forest #ai #python #flashcard 