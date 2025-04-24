### created: 21-11-2023
---
### Concept  
Explain the core idea in your own words. What is it?

What is the concept of K Fold Cross Validation::K-Fold Cross Validation is a resampling technique used to evaluate machine learning models by splitting the dataset into K equal parts (folds), then training the model K times, each time using a different fold as the validation set and the remaining K-1 folds as the training set. The model's performance is then averaged across all K iterations to provide a more robust estimate of its predictive ability on unseen data, reducing the risk of overfitting to a specific train-test split.

---
### Context  
Where and when is it used? Why is it important?

In what context is K Fold Cross Validation typically applied::K-Fold Cross Validation is applied when evaluating machine learning models, especially with limited data, when tuning hyperparameters, when comparing different algorithms, and when building ensemble models. It's important because it provides a more reliable performance estimate than a single train-test split, reduces the risk of overfitting during model selection and hyperparameter tuning, makes efficient use of limited data, helps detect model instability or high variance, and enables statistical comparisons between different models.

---
### Connection  
Link this to related concepts, building blocks, or prerequisites.

- [[What_is_Machine_Learning]]
- [[Training_and_Testing_Data]]
- [[Hyper_parameter_Tuning]]
- [[Bias_vs_Variance]]
- [[Decision_Tree]]
- [[Random_Forest]]
- [[Support_Vector_Machine]]

What concepts are connected to K Fold Cross Validation::[[What_is_Machine_Learning]], [[Training_and_Testing_Data]], [[Hyper_parameter_Tuning]], [[Bias_vs_Variance]], [[Decision_Tree]], [[Random_Forest]], [[Support_Vector_Machine]]

---
### Concrete Example  
Provide a practical example (code snippet, diagram, equation, or analogy).

```python
# K-Fold Cross Validation example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"Dataset shape: {X.shape}")
print(f"Classes: {target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Part 1: Basic K-Fold Cross Validation
print("\n=== Basic K-Fold Cross Validation ===")

# Define the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Lists to store results
fold_accuracies = []
fold_predictions = []
all_y_true = []

# Manual K-Fold implementation (for illustration)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
    
    # Train the model
    model.fit(X_train_fold, y_train_fold)
    
    # Make predictions
    y_pred_fold = model.predict(X_val_fold)
    fold_predictions.extend(y_pred_fold)
    all_y_true.extend(y_val_fold)
    
    # Calculate and store accuracy
    accuracy = accuracy_score(y_val_fold, y_pred_fold)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

# Calculate overall metrics
overall_accuracy = accuracy_score(all_y_true, fold_predictions)
print(f"\nAverage Accuracy across {k} folds: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Part 2: Using scikit-learn's cross_val_score (simpler approach)
print("\n=== Using scikit-learn's cross_val_score ===")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Evaluate each model using cross-validation
for name, model in models.items():
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=k, scoring='accuracy')
    
    print(f"{name} - Mean Accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")

# Part 3: Multiple metrics with cross_validate
print("\n=== Multiple Metrics with cross_validate ===")

# Define multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro',
    'roc_auc': 'roc_auc'
}

# Create a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline = Pipeline([('scaler', StandardScaler()), ('model', rf)])

# Perform cross-validation with multiple metrics
cv_results = cross_validate(pipeline, X, y, cv=k, scoring=scoring)

# Print results
for metric in scoring.keys():
    mean_score = cv_results[f'test_{metric}'].mean()
    std_score = cv_results[f'test_{metric}'].std()
    print(f"{metric.capitalize()}: {mean_score:.4f} ± {std_score:.4f}")

# Part 4: Hyperparameter tuning with GridSearchCV (uses cross-validation internally)
print("\n=== Hyperparameter Tuning with GridSearchCV ===")

# Define the parameter grid for Random Forest
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

# Create the pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Create GridSearchCV
grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=k,
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)

# Fit GridSearchCV
grid_search.fit(X, y)

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Part 5: Visualizations
# Visualize cross-validation results across models
plt.figure(figsize=(15, 10))

# Plot 1: Compare model performances
plt.subplot(2, 2, 1)
model_names = list(models.keys())
model_scores = []

for name, model in models.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    scores = cross_val_score(pipeline, X, y, cv=k, scoring='accuracy')
    model_scores.append(scores)

plt.boxplot(model_scores, labels=model_names)
plt.title('Model Performance Comparison')
plt.ylabel('Accuracy')
plt.grid(True, axis='y')

# Plot 2: Visualize fold predictions vs actual for a single model
plt.subplot(2, 2, 2)
# Assuming fold_predictions and all_y_true from Part 1
confusion_data = pd.DataFrame({
    'Actual': all_y_true,
    'Predicted': fold_predictions
})
confusion_counts = pd.crosstab(confusion_data['Actual'], confusion_data['Predicted'], normalize='index')
sns.heatmap(confusion_counts, annot=True, cmap='Blues', cbar=False, fmt='.2f',
           xticklabels=target_names, yticklabels=target_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 3: Learning curve (train vs validation scores vs training set size)
from sklearn.model_selection import learning_curve

plt.subplot(2, 2, 3)
train_sizes, train_scores, test_scores = learning_curve(
    Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))]),
    X, y, cv=k, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.grid()

# Plot 4: Fold-by-fold comparison for multiple models
plt.subplot(2, 2, 4)
fold_data = []

for name, model in models.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_train, y_train)
        val_score = accuracy_score(y_val, pipeline.predict(X_val))
        
        fold_data.append({
            'Model': name,
            'Fold': fold_idx + 1,
            'Accuracy': val_score
        })

fold_df = pd.DataFrame(fold_data)
sns.barplot(x='Fold', y='Accuracy', hue='Model', data=fold_df)
plt.title('Model Performance by Fold')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  # Adjust as needed for your data
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
```

**Common K-Fold Cross Validation Variations:**

1. **Standard K-Fold CV**: Splits data into k equal folds, uses each fold once as the validation set.

2. **Stratified K-Fold CV**: Ensures that each fold maintains the same class distribution as the original dataset, crucial for imbalanced datasets.

3. **Leave-One-Out CV (LOOCV)**: Special case where k equals the number of samples, each sample is used once as a validation set.

4. **Group K-Fold CV**: Ensures that samples from the same group (e.g., patients from the same hospital) aren't split across training and validation sets.

5. **Time Series CV**: For temporal data, validation sets are always chronologically after the training sets, preserving temporal order.

---
### Iterative Thinking
Reflect to deepen your learning.

What's one thing I'm still unsure about?
How to determine the optimal number of folds (k) for different dataset sizes and model types, balancing between computational costs and reliable performance estimates?

How can I apply this to a real project or problem?
I could use k-fold cross-validation to evaluate different classifiers for a medical diagnosis system, ensuring the selected model generalizes well across different patient subgroups and providing reliable confidence intervals for performance metrics.

What's a common misconception or edge case?
A common misconception is that larger k values are always better. While higher k (e.g., k=10) provides more training data and potentially less biased estimates, it also increases computational cost and can lead to higher variance in performance estimates. For small datasets, high k values might lead to validation sets that are too small to provide reliable estimates.

The key idea behind K Fold Cross Validation is {{repeatedly partitioning data into training and validation sets to obtain a more reliable estimate of model performance on unseen data}}.

---
##### Tags

#ai/K_Fold_Cross_Validation #ai #python #flashcard 