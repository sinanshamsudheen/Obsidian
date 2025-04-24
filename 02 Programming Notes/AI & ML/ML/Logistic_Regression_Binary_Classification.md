### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of Logistic Regression Binary Classification::Logistic Regression for Binary Classification is a supervised learning algorithm that predicts the probability of an observation belonging to one of two classes. Despite its name, it's a classification algorithm, not a regression algorithm. It uses the logistic (sigmoid) function to model the probability that an input belongs to the positive class, and classifies instances based on whether this probability exceeds a threshold (typically 0.5).

---
### Context  
Where and when is it used? Why is it important?

In what context is Logistic Regression Binary Classification typically applied::Logistic Regression is applied in scenarios requiring binary outcomes, such as spam detection (spam/not spam), disease diagnosis (diseased/healthy), customer churn prediction (will churn/won't churn), and credit approval (approve/deny). It's important because it's interpretable (coefficients indicate feature importance), computationally efficient, provides probability estimates rather than just classifications, handles both numerical and categorical features, and serves as a baseline for more complex classification models.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Linear_Regression_Multiple_Variables]]
- [[Gradient_Descent_and_Cost_Function]]
- [[Logistic_Regression_Multiclass_Classification]]
- [[Training_and_Testing_Data]]
- [[L1_and_L2_Regularization]]
- [[Dummy_Variables_One_Hot_Encoding]]

What concepts are connected to Logistic Regression Binary Classification::[[What_is_Machine_Learning]], [[Linear_Regression_Multiple_Variables]], [[Gradient_Descent_and_Cost_Function]], [[Logistic_Regression_Multiclass_Classification]], [[Training_and_Testing_Data]], [[L1_and_L2_Regularization]], [[Dummy_Variables_One_Hot_Encoding]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# Logistic Regression for Binary Classification example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns

# Generate a synthetic dataset for binary classification
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 2)  # Two features for easy visualization

# Create two clusters with overlap
class0_indices = np.random.choice(n_samples, int(n_samples * 0.6), replace=False)
class1_indices = np.array(list(set(range(n_samples)) - set(class0_indices)))

# Shift class 1 points to create separation
X[class1_indices, 0] += 2
X[class1_indices, 1] += 1

# Create binary labels
y = np.zeros(n_samples)
y[class1_indices] = 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Get predictions and probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Model coefficients
print("\nModel Coefficients:")
for i, coef in enumerate(model.coef_[0]):
    print(f"Feature {i+1}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Explanation of the logistic function and how predictions are made
# The logistic function: p(x) = 1 / (1 + e^(-z)) where z = b0 + b1*x1 + b2*x2 + ...
z = X_test_scaled.dot(model.coef_[0]) + model.intercept_[0]
p = 1 / (1 + np.exp(-z))

print("\nExample prediction:")
example_idx = 0
print(f"Features: {X_test_scaled[example_idx]}")
print(f"z = {z[example_idx]:.4f}")
print(f"p(y=1) = 1 / (1 + e^(-z)) = {p[example_idx]:.4f}")
print(f"Predicted class: {1 if p[example_idx] >= 0.5 else 0}")
print(f"Actual class: {int(y_test[example_idx])}")

# Visualizations
# Plot decision boundary
plt.figure(figsize=(12, 10))

# Plot 1: Data with decision boundary
plt.subplot(2, 2, 1)
# Create a meshgrid to visualize decision boundary
h = 0.02  # Step size
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, s=50, 
           cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.7)
plt.title('Decision Boundary')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')

# Plot 2: Confusion Matrix
plt.subplot(2, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
           xticklabels=['Class 0', 'Class 1'],
           yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 3: ROC Curve
plt.subplot(2, 2, 3)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')

# Plot 4: Probability Distribution
plt.subplot(2, 2, 4)
df_results = pd.DataFrame({
    'True Class': y_test,
    'Probability': y_prob
})
sns.histplot(data=df_results, x='Probability', hue='True Class', bins=20, alpha=0.6)
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
plt.title('Probability Distribution')
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()
plt.show()
```

The equation for logistic regression:
p(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))

Where:
- p(y=1|x) is the probability that the dependent variable equals 1 given x
- β₀ is the intercept
- β₁, β₂, ..., βₙ are the regression coefficients
- x₁, x₂, ..., xₙ are the independent variables

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to properly handle highly imbalanced datasets in logistic regression, where one class is much more frequent than the other?

How can I apply this to a real project or problem?
I could use logistic regression to develop a model that predicts whether a customer will respond to a marketing campaign based on demographic data and past purchase behavior, allowing for more targeted marketing efforts.

What's a common misconception or edge case?
A common misconception is that logistic regression always works well with the default 0.5 threshold. In practice, the optimal threshold often depends on the relative costs of false positives versus false negatives, and needs to be chosen based on the specific business context.

The key idea behind Logistic Regression Binary Classification is {{modeling the probability of binary outcomes using the sigmoid function transformation}}.

---
##### Tags

#ai/Logistic_Regression_Binary_Classification #ai #python #flashcard 
 