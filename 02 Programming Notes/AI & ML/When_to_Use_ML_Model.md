# 🧠 When to Use a Particular Machine Learning Model

This Obsidian note helps you decide **which ML model to use** depending on your data type, objective, interpretability need, and scalability.

---

## 🔍 Classification

| Use Case | Recommended Models | Notes |
|----------|--------------------|-------|
| Binary Classification | Logistic Regression, Random Forest, XGBoost | Use Logistic for interpretability, XGBoost for performance |
| Multiclass Classification | Random Forest, XGBoost, LightGBM, Neural Networks | Tree-based for tabular, DNNs for image/text |
| Text Classification | Naive Bayes, Logistic Regression, BERT, DistilBERT | Start with TF-IDF + LR, move to transformers for semantics |
| Imbalanced Data | XGBoost, LightGBM, CatBoost + class weights/SMOTE | Use built-in class weighting or resampling |

---

## 📈 Regression

| Use Case | Recommended Models | Notes |
|----------|--------------------|-------|
| Linear Relationship | Linear Regression, Ridge, Lasso | Good baseline models |
| Non-linear Patterns | Random Forest, XGBoost, SVR | Use if linear models underperform |
| High-Dimensional Features | Lasso, ElasticNet, XGBoost | Lasso for feature selection |
| Time-Series Forecasting | ARIMA, Prophet, LSTM | Use ARIMA/Prophet for simple trends, LSTM for deep patterns |

---

## 📊 Clustering

| Use Case | Recommended Models | Notes |
|----------|--------------------|-------|
| General-purpose clustering | KMeans, DBSCAN, Agglomerative | KMeans is fast, DBSCAN for non-globular clusters |
| High-dimensional data | Spectral Clustering, PCA + KMeans | Reduce dimensions before clustering |
| Unknown number of clusters | DBSCAN, Gaussian Mixture Models | DBSCAN infers clusters automatically |

---

## ⚠️ Anomaly Detection

| Use Case | Recommended Models | Notes |
|----------|--------------------|-------|
| Unsupervised | Isolation Forest, One-Class SVM, Autoencoder | Isolation Forest for speed, Autoencoders for depth |
| Time-series anomalies | LSTM, Prophet, Twitter AnomalyDetection | LSTM for sequence modeling |
| Real-time detection | Isolation Forest, Streaming KMeans | Use with Kafka/Spark for SOCs |

---

## 🧬 NLP Tasks

| Task | Recommended Models | Notes |
|------|--------------------|-------|
| Sentiment Analysis | Logistic Regression, DistilBERT | Use Logistic for TF-IDF baseline |
| Named Entity Recognition | spaCy, BERT, Flair | Pretrained models save time |
| Text Generation | GPT-2, T5 | Requires fine-tuning for domain-specific tasks |
| Translation | mBART, MarianMT | Use pretrained multilingual models |

---

## 🧠 Deep Learning Scenarios

| Task | Model | Notes |
|------|-------|-------|
| Image Classification | CNN, ResNet, EfficientNet | Transfer learning works well |
| Sequence Modeling | RNN, LSTM, GRU, Transformers | Transformers dominate modern NLP |
| Tabular + Complex Patterns | TabNet, FT-Transformer | Good for hybrid scenarios |

---

## 🎯 Model Selection Criteria

| Criterion | Considerations |
|----------|----------------|
| Accuracy | XGBoost, Neural Networks often best |
| Interpretability | Logistic Regression, Decision Trees, SHAP-compatible models |
| Speed | Logistic Regression, Random Forest |
| Real-time Inference | DistilBERT, LightGBM, Isolation Forest |
| Data Size | Tree models for small/medium, DL for large |
| Feature Engineering | AutoML if manual effort is a bottleneck |

---

## 🔧 Tools

- `scikit-learn`: Classical ML
- `xgboost`, `lightgbm`, `catboost`: Gradient boosting
- `transformers`: NLP
- `tensorflow`, `pytorch`: Deep learning
- `h2o`, `AutoSklearn`, `TPOT`: AutoML

---

## 📌 Tips

- Always **start simple** (Logistic, Decision Tree) before deep models.
- Use **cross-validation** for evaluation.
- Monitor **overfitting** and use regularization if needed.
- For imbalanced data, check precision/recall not just accuracy.
