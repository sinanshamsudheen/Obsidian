### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Decision Tree::A Decision Tree is a tree-structured model where internal nodes represent features, branches represent decision rules, and leaf nodes represent outcomes or predictions. It recursively splits the data based on feature values to create homogeneous subsets, using metrics like Gini impurity or information gain to determine the best splits. The model makes predictions by navigating from the root to a leaf node following decision rules.

---
### Context  
Where and when is it used? Why is it important?

In what context is Decision Tree typically applied::Decision Trees are applied in classification and regression problems across domains like finance (credit scoring), healthcare (diagnosis), marketing (customer segmentation), and resource management. They're used when interpretability is crucial, when handling mixed data types, or as building blocks for ensemble methods. They're important because they're intuitive and easily explainable (can visualize the decision process), handle both numerical and categorical features without preprocessing, automatically perform feature selection, can model non-linear relationships, and capture feature interactions.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Training_and_Testing_Data]]
- [[Random_Forest]]
- [[Ensemble_Learning_Bagging]]
- [[K_Fold_Cross_Validation]]
- [[Hyper_parameter_Tuning]]
- [[Bias_vs_Variance]]

What concepts are connected to Decision Tree::[[What_is_Machine_Learning]], [[Training_and_Testing_Data]], [[Random_Forest]], [[Ensemble_Learning_Bagging]], [[K_Fold_Cross_Validation]], [[Hyper_parameter_Tuning]], [[Bias_vs_Variance]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Decision Tree example for both classification and regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Part 1: Decision Tree for Classification
print("=== Decision Tree Classification ===")
# Load Iris dataset
iris = load_iris()
X_class = iris.data
y_class = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
)

# Create and train the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_class, y_train_class)

# Make predictions
y_pred_class = dt_classifier.predict(X_test_class)

# Evaluate the model
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Classification Accuracy: {accuracy:.4f}")

# Feature importance
class_importances = dt_classifier.feature_importances_
sorted_idx = np.argsort(class_importances)
print("\nFeature Importance (Classification):")
for i in sorted_idx[::-1]:
    print(f"  {feature_names[i]}: {class_importances[i]:.4f}")

# Part 2: Decision Tree for Regression
print("\n=== Decision Tree Regression ===")
# Load California Housing dataset (just a sample for demonstration)
housing = fetch_california_housing()
X_reg = housing.data[:500]
y_reg = housing.target[:500]
reg_feature_names = housing.feature_names

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Scale features for regression
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Create and train the model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_reg_scaled, y_train_reg)

# Make predictions
y_pred_reg = dt_regressor.predict(X_test_reg_scaled)

# Evaluate the model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression Mean Squared Error: {mse:.4f}")
print(f"Regression R² Score: {r2:.4f}")

# Feature importance
reg_importances = dt_regressor.feature_importances_
sorted_idx = np.argsort(reg_importances)
print("\nFeature Importance (Regression):")
for i in sorted_idx[::-1]:
    print(f"  {reg_feature_names[i]}: {reg_importances[i]:.4f}")

# Part 3: Visualizations
plt.figure(figsize=(20, 12))

# Plot 1: Decision Tree Classifier Visualization
plt.subplot(2, 2, 1)
plot_tree(dt_classifier, feature_names=feature_names, class_names=target_names, 
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree Classifier')

# Plot 2: Classification Feature Importance
plt.subplot(2, 2, 2)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': dt_classifier.feature_importances_
})
importance_df = importance_df.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Classification)')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Plot 3: Decision Boundaries (for two features only)
plt.subplot(2, 2, 3)
feature_idx1, feature_idx2 = 2, 3  # Petal length and width

# Create a mesh grid
h = 0.02
x_min, x_max = X_class[:, feature_idx1].min() - 1, X_class[:, feature_idx1].max() + 1
y_min, y_max = X_class[:, feature_idx2].min() - 1, X_class[:, feature_idx2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Create a simplified dataset with only the two selected features
X_simplified = X_class[:, [feature_idx1, feature_idx2]]
X_train_simplified, X_test_simplified, y_train_simplified, y_test_simplified = train_test_split(
    X_simplified, y_class, test_size=0.3, random_state=42, stratify=y_class
)

dt_simple = DecisionTreeClassifier(random_state=42)
dt_simple.fit(X_train_simplified, y_train_simplified)

# Predict on mesh grid points
Z = dt_simple.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_test_simplified[:, 0], X_test_simplified[:, 1], c=y_test_simplified, 
           cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel(feature_names[feature_idx1])
plt.ylabel(feature_names[feature_idx2])
plt.title('Decision Boundaries on Test Data')

# Plot 4: Effect of Tree Depth on Overfitting
plt.subplot(2, 2, 4)
depths = range(1, 15)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_class, y_train_class)
    train_scores.append(accuracy_score(y_train_class, dt.predict(X_train_class)))
    test_scores.append(accuracy_score(y_test_class, dt.predict(X_test_class)))

plt.plot(depths, train_scores, 'o-', color='blue', label='Training accuracy')
plt.plot(depths, test_scores, 'o-', color='red', label='Testing accuracy')
plt.xlabel('Max Tree Depth')
plt.ylabel('Accuracy')
plt.title('Effect of Tree Depth on Performance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Part 4: Demonstrate pruning with max_depth and min_samples_leaf
print("\n=== Decision Tree Pruning ===")
pruned_dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
pruned_dt.fit(X_train_class, y_train_class)
pruned_pred = pruned_dt.predict(X_test_class)
pruned_accuracy = accuracy_score(y_test_class, pruned_pred)

print(f"Unpruned tree accuracy: {accuracy:.4f}")
print(f"Pruned tree accuracy (max_depth=3, min_samples_leaf=5): {pruned_accuracy:.4f}")
print(f"Unpruned tree depth: {dt_classifier.get_depth()}")
print(f"Pruned tree depth: {pruned_dt.get_depth()}")
print(f"Unpruned tree node count: {dt_classifier.tree_.node_count}")
print(f"Pruned tree node count: {pruned_dt.tree_.node_count}")

# Visualize pruned tree
plt.figure(figsize=(15, 10))
plot_tree(pruned_dt, feature_names=feature_names, class_names=target_names, 
          filled=True, rounded=True, fontsize=12)
plt.title('Pruned Decision Tree (max_depth=3, min_samples_leaf=5)')
plt.tight_layout()
plt.show()
```

Decision Tree Split Criteria:

1. Gini Impurity: 
   - Gini = 1 - Σ(pᵢ²)
   - Where pᵢ is the probability of class i in the node

2. Entropy/Information Gain:
   - Entropy = -Σ(pᵢ × log₂(pᵢ))
   - Information Gain = Entropy(parent) - [weighted average of Entropy(children)]

3. Mean Squared Error (for regression):
   - MSE = (1/n) × Σ(yᵢ - ȳ)²
   - Where ȳ is the mean target value in the node

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to determine the optimal balance between tree depth and generalization performance without extensive hyperparameter tuning or cross-validation?

How can I apply this to a real project or problem?
I could use a decision tree to create a loan approval system that makes transparent decisions based on applicant attributes like income, credit history, and employment status, providing clear explanations for approvals or rejections.

What's a common misconception or edge case?
A common misconception is that deeper trees always lead to better performance. In reality, unpruned trees often overfit to the training data, capturing noise rather than patterns, which is why techniques like pruning, setting maximum depth, or ensemble methods are important for improving generalization.

The key idea behind Decision Tree is {{hierarchical partitioning of data based on features to create simple, interpretable decision rules}}.

---
##### Tags

#ai/Decision_Tree #ai #python #flashcard 