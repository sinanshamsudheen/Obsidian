# Data Leak Prevention (DLP) using BERT + CatBoost - SOC Intern Handbook

**Approach:** Data Leak Prevention (DLP) using BERT for Semantic Context + CatBoost for Behavioral Fusion

---

## 🔹 Section 1: Title & Problem Statement

### What Problem This Approach Solves

Modern data exfiltration attacks have evolved beyond simple file copying. Sophisticated insider threats and advanced attackers use multi-channel, context-aware techniques to steal sensitive data while evading traditional DLP systems. They employ:

- **Semantic Obfuscation**: Referring to "Q2 numbers" instead of "financial reports"
- **Multi-Vector Exfiltration**: USB + cloud upload + messaging within short timeframes
- **Contextual Camouflage**: Disguising sensitive content as normal business communications
- **Compression/Encoding**: ZIP files, base64 encoding, or encrypted containers to hide content
- **Behavioral Blending**: Timing attacks during normal business hours or legitimate-looking activities

### Why Traditional Methods Fail

- **Rule-Based DLP**: Cannot understand semantic meaning - fails when attackers use synonyms or contextual references
- **Signature Detection**: Easily bypassed with compression, encryption, or format conversion
- **Single-Channel Focus**: Misses coordinated attacks across multiple exfiltration vectors
- **No Behavioral Context**: Treats all users equally, ignoring individual behavior patterns and risk profiles
- **High False Positives**: Over-aggressive rules block legitimate business activities, leading to policy bypass

### Why This ML Method is Preferred

The hybrid BERT + CatBoost approach provides comprehensive protection by:

**BERT (Semantic Understanding):**

- **Deep Language Comprehension**: Understands intent and context, not just keywords
- **Synonym Detection**: Recognizes "customer list" = "client database" = "contact roster"
- **Contextual Awareness**: Distinguishes between legitimate business use and suspicious contexts
- **Obfuscation Resistance**: Handles paraphrasing, euphemisms, and indirect references

**CatBoost (Behavioral Analysis):**

- **Multi-Channel Correlation**: Detects coordinated activities across different systems
- **Temporal Pattern Recognition**: Identifies unusual timing, volume, or frequency patterns
- **User Behavioral Baselines**: Personalizes detection based on individual user patterns
- **Categorical Feature Handling**: Natively processes device types, applications, and destinations

**Fusion Benefits:**

- **Reduced False Positives**: Behavioral context validates semantic alerts
- **Enhanced Detection**: Catches both content-based and behavior-based threats
- **Adaptive Learning**: Continuously improves understanding of both content and behavior patterns

---

## 🔹 Section 2: Detailed Explanation of the Approach

### Step-by-Step Breakdown

#### 1. Multi-Channel Data Ingestion

**Content Sources:**

- Email systems (Exchange, Gmail, custom SMTP)
- Messaging platforms (Slack, Teams, WhatsApp Business)
- Cloud storage uploads (OneDrive, Google Drive, Dropbox)
- File transfer logs (SFTP, HTTP uploads, P2P)
- Document management systems (SharePoint, Confluence)

**Behavioral Sources:**

- Endpoint agent logs (file access, USB events, screen captures)
- Network proxy logs (destinations, volumes, timing)
- Application usage logs (copy/paste events, print jobs)
- Authentication logs (login patterns, device changes)
- VPN and remote access logs

#### 2. Preprocessing Pipeline

**Content Preprocessing:**

```python
# BERT tokenization pipeline
def preprocess_content(text):
    # Handle multiple formats
    if is_html(text):
        text = html_to_text(text)
    elif is_pdf(text):
        text = pdf_to_text(text)
    
    # Unicode normalization for obfuscation detection
    text = unicodedata.normalize('NFKC', text)
    
    # BERT tokenization with special tokens
    tokens = bert_tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return tokens
```

**Behavioral Data Aggregation:**

- Time-windowed aggregation (5-minute, 1-hour, daily buckets)
- Cross-system activity correlation using user IDs and timestamps
- Feature engineering for volume, frequency, and pattern metrics

#### 3. BERT Semantic Analysis Engine

**Fine-Tuning Strategy:**

```python
# Domain-specific BERT fine-tuning
class DLPBertClassifier(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
```

**Semantic Feature Extraction:**

- **Contextual Embeddings**: 768-dimensional vectors capturing semantic meaning
- **Similarity Scoring**: Cosine similarity against known sensitive document embeddings
- **Topic Classification**: Categories like "Financial", "PII", "IP", "HR", "Legal"
- **Sentiment Analysis**: Unusual emotional context that might indicate malicious intent
- **Entity Recognition**: Automatic detection of names, SSNs, credit cards, IP addresses

#### 4. CatBoost Behavioral Analysis

**Feature Engineering Pipeline:**

```python
# Behavioral feature extraction
def extract_behavioral_features(user_id, time_window):
    features = {
        # Volume features
        'total_data_volume': get_volume_by_user(user_id, time_window),
        'volume_deviation': volume - user_baseline_volume,
        
        # Timing features  
        'hour_of_day': current_hour,
        'is_weekend': is_weekend(current_time),
        'outside_business_hours': not in_business_hours(current_time),
        
        # Channel features
        'unique_channels_used': count_unique_channels(user_id, time_window),
        'high_risk_destinations': count_external_destinations(user_id, time_window),
        
        # Pattern features
        'activity_burst': detect_activity_spike(user_id, time_window),
        'new_device_usage': is_new_device(user_id, current_device),
        
        # Risk context
        'user_risk_tier': get_user_risk_level(user_id),
        'recent_access_escalation': check_privilege_changes(user_id, 30)
    }
    return features
```

**CatBoost Model Configuration:**

```python
# Optimized CatBoost for DLP
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 8,
    'l2_leaf_reg': 10,
    'random_seed': 42,
    'class_weights': [1, 3],  # Handle imbalanced data
    'cat_features': ['device_type', 'application', 'destination_category'],
    'eval_metric': 'AUC',
    'use_best_model': True,
    'early_stopping_rounds': 50
}
```

#### 5. Fusion Architecture

**Multi-Modal Integration:**

```python
class DLPFusionModel:
    def __init__(self):
        self.bert_model = load_fine_tuned_bert()
        self.catboost_model = load_catboost_model()
        self.fusion_weights = {'semantic': 0.6, 'behavioral': 0.4}
    
    def predict_risk(self, content, behavioral_features):
        # Semantic analysis
        semantic_embedding = self.bert_model.encode(content)
        semantic_risk = self.calculate_semantic_risk(semantic_embedding)
        
        # Behavioral analysis  
        behavioral_risk = self.catboost_model.predict_proba(behavioral_features)[1]
        
        # Weighted fusion
        final_risk = (
            self.fusion_weights['semantic'] * semantic_risk +
            self.fusion_weights['behavioral'] * behavioral_risk
        )
        
        return {
            'risk_score': final_risk,
            'semantic_score': semantic_risk,
            'behavioral_score': behavioral_risk,
            'explanation': self.generate_explanation()
        }
```

### Real-World Significance

#### Semantic Understanding Impact

- **Context Preservation**: Understands "the Johnson project files" refers to a specific client engagement
- **Intent Detection**: Distinguishes between "sharing financial projections for meeting" vs. "sending revenue data to personal email"
- **Language Evolution**: Adapts to new terminology and business language changes

#### Behavioral Analysis Impact

- **User-Specific Baselines**: Recognizes that midnight file access is normal for night-shift workers but suspicious for day-shift employees
- **Multi-Channel Correlation**: Connects USB insertion + large file copy + external email within 10 minutes
- **Temporal Context**: Flags unusual patterns like data access spikes before employee departures

---

## 🔹 Section 3: Interview Q&A (SOC + ML)

### Q1: Why combine BERT with CatBoost instead of using a single end-to-end model?

**Answer**: The combination leverages the strengths of both specialized architectures:

- **BERT excels at semantic understanding** - it captures language nuances, context, and meaning that traditional NLP approaches miss
- **CatBoost excels at structured behavioral data** - it handles categorical features, missing values, and complex feature interactions natively
- **Different data types require different approaches** - unstructured text (emails, documents) needs transformer attention mechanisms, while structured behavioral data (timestamps, volumes, categories) needs gradient boosting
- **Interpretability**: Separate models allow analysts to understand both "what content was risky" and "what behavior was suspicious"
- **Performance**: Specialized models often outperform general end-to-end approaches for multi-modal problems

### Q2: How does the system handle sophisticated adversarial tactics like semantic obfuscation?

**Answer**: The system has multiple layers of defense against advanced evasion:

- **BERT's Contextual Understanding**: Recognizes paraphrasing like "Q2 numbers" → "financial reports" through semantic similarity
- **Embedding Space Analysis**: Similar concepts cluster together in BERT's embedding space, making synonyms detectable
- **Behavioral Correlation**: Even if content is obfuscated, unusual behavioral patterns (timing, volume, channels) still trigger alerts
- **Multi-Vector Detection**: Attackers might obfuscate content but can't easily disguise coordinated multi-channel activities
- **Continuous Learning**: Regular retraining on new obfuscation techniques discovered through red team exercises
- **Entropy Analysis**: Compressed or encoded content exhibits different statistical properties that CatBoost can detect

### Q3: How do you ensure real-time performance with computationally expensive BERT inference?

**Answer**: Several optimization strategies enable real-time deployment:

- **Model Distillation**: Use DistilBERT or TinyBERT for 60-80% of full BERT performance with 4x speed improvement
- **Caching Strategy**: Cache BERT embeddings for frequently accessed documents and templates
- **Tiered Processing**: Fast rule-based pre-filtering, then BERT analysis only for suspicious content
- **Batch Processing**: Group multiple documents for efficient GPU utilization
- **Edge Deployment**: Deploy smaller models on endpoints for immediate response, full analysis in cloud
- **Asynchronous Architecture**: Immediate blocking based on behavioral signals, semantic analysis for forensics
- **Hardware Optimization**: GPU inference for BERT, CPU for CatBoost, optimized for 300-600ms total latency

### Q4: How does the system maintain privacy while analyzing sensitive content?

**Answer**: Privacy preservation is built into the architecture:

- **Federated Learning**: BERT fine-tuning occurs locally on departmental data, only model updates shared
- **Embedding-Only Storage**: Store semantic embeddings, not raw content, for pattern analysis
- **Differential Privacy**: Add noise for demographic aggregations and trend analysis
- **Selective Logging**: Only high-risk events above threshold are logged with full context
- **Content Hashing**: Use perceptual hashing for image-based content to detect duplicates without storage
- **Role-Based Access**: Different privacy levels based on analyst roles and investigation scope
- **Audit Trails**: Complete logging of who accessed what sensitive data during investigations

### Q5: How do you prevent legitimate business activities from being flagged as data leaks?

**Answer**: False positive reduction is critical for business adoption:

- **User Behavioral Baselines**: Learn individual patterns - what's normal for each user/role
- **Business Context Integration**: Integrate with HR systems to understand job roles, project assignments, approved external collaborations
- **Temporal Context**: Consider business cycles, deadlines, and approved project timelines
- **Approval Workflows**: Integrate with existing business approval processes for planned data sharing
- **Whitelist Management**: Maintain approved external destinations and business partner domains
- **Confidence Thresholds**: Use adaptive thresholds - higher for executives, lower for high-risk roles
- **Human-in-the-Loop**: Analyst review for edge cases with business impact assessment

### Q6: How does the system adapt to new attack techniques or insider threat patterns?

**Answer**: Continuous adaptation through multiple mechanisms:

- **Adversarial Training**: Regular red team exercises provide new attack samples for model improvement
- **Transfer Learning**: Leverage threat intelligence and attack patterns from security community
- **Active Learning**: Prioritize analyst feedback on challenging cases for targeted model improvement
- **Ensemble Evolution**: Regularly evaluate and update model components based on performance
- **Feature Engineering Pipeline**: Automated discovery of new behavioral patterns and content features
- **Threat Intelligence Integration**: Incorporate IOCs and TTPs from external feeds
- **A/B Testing**: Gradual rollout of model updates with performance comparison

### Q7: What are the key metrics for measuring DLP effectiveness in a SOC environment?

**Answer**: Comprehensive metrics across multiple dimensions:

**Detection Performance:**

- **True Positive Rate**: Percentage of actual data leaks detected (target: >90%)
- **False Positive Rate**: Legitimate activities incorrectly flagged (target: <5%)
- **Time to Detection**: Average time from leak attempt to alert (target: <5 minutes)

**Operational Impact:**

- **Analyst Workload**: Hours spent on DLP investigations vs. other security tasks
- **Alert Quality**: Percentage of alerts that result in actual incidents or policy violations
- **Investigation Time**: Average time to resolve DLP alerts (target: <30 minutes per alert)

**Business Impact:**

- **Data Loss Prevention**: Estimated value of prevented data exposure
- **Compliance Adherence**: Percentage of regulatory requirements met (GDPR, HIPAA, etc.)
- **User Productivity**: Minimal impact on legitimate business workflows
- **Risk Reduction**: Decrease in data breach probability and potential impact

---

## 🔹 Section 4: Deployment Readiness

### Latency Expectations

#### Component-Level Performance

- **BERT Inference**: 100-300ms per document (depending on length and hardware)
- **CatBoost Prediction**: 5-20ms per behavioral feature set
- **Feature Extraction**: 50-100ms for behavioral aggregation
- **Total End-to-End**: 300-600ms for complete analysis

#### Optimization Targets

- **Real-time Decisions**: <1 second for block/allow determinations
- **Batch Processing**: 1000+ documents/minute for historical analysis
- **Streaming Analysis**: Handle 10K+ events/second with proper scaling

### Integration Architecture

#### Inline Deployment Options

```python
# DLP Gateway Integration
class DLPGateway:
    def __init__(self):
        self.bert_model = load_model('dlp-bert')
        self.catboost_model = load_model('dlp-catboost')
        self.policy_engine = PolicyEngine()
    
    def evaluate_transfer(self, content, metadata, user_context):
        # Quick behavioral pre-screening
        behavioral_risk = self.catboost_model.predict(
            extract_behavioral_features(user_context)
        )
        
        # If behavioral risk is low, allow with minimal processing
        if behavioral_risk < 0.3:
            return {'action': 'allow', 'risk_score': behavioral_risk}
        
        # Full semantic analysis for suspicious behavior
        semantic_risk = self.bert_model.analyze_content(content)
        
        # Policy decision
        return self.policy_engine.make_decision(
            semantic_risk, behavioral_risk, user_context
        )
```

#### SIEM Integration Patterns

```python
# Splunk Integration Example
def send_to_splunk(dlp_event):
    splunk_event = {
        'timestamp': dlp_event['timestamp'],
        'user': dlp_event['user_id'],
        'risk_score': dlp_event['risk_score'],
        'semantic_score': dlp_event['semantic_score'],
        'behavioral_score': dlp_event['behavioral_score'],
        'action_taken': dlp_event['action'],
        'content_category': dlp_event['content_type'],
        'destination': dlp_event['destination'],
        'explanation': dlp_event['explanation']
    }
    
    splunk_client.index('dlp_events').submit(
        json.dumps(splunk_event),
        sourcetype='dlp:risk_assessment'
    )
```

### Monitoring Dashboards

#### Real-Time Operations Dashboard

- **Risk Score Distribution**: Live histogram of current risk assessments
- **Alert Volume**: Trends in DLP alerts by hour/day with seasonal baselines
- **Top Risk Users**: Users with highest cumulative risk scores
- **Content Categories**: Breakdown of sensitive content types being shared
- **Channel Analysis**: Risk distribution across different exfiltration channels

#### Model Performance Dashboard

- **Inference Latency**: P95/P99 latency metrics for BERT and CatBoost
- **Model Drift Detection**: Distribution shifts in features and predictions
- **False Positive Trends**: Weekly false positive rates with analyst feedback
- **Coverage Metrics**: Percentage of data flows monitored vs. total organization traffic

#### Business Impact Dashboard

- **Prevented Loss Estimation**: Estimated value of blocked data transfers
- **Compliance Status**: Current compliance with regulatory requirements
- **Policy Effectiveness**: Success rates of different DLP policies
- **User Training Needs**: Users with high false positive rates needing education

---

## 🔹 Section 5: Optimization Tips

### Model Tuning Strategies

#### BERT Fine-Tuning Optimization

```python
# Optimized training configuration
training_args = TrainingArguments(
    output_dir='./dlp-bert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    # Mixed precision for faster training
    fp16=True,
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing=True
)
```

#### CatBoost Hyperparameter Optimization

```python
# Bayesian optimization for CatBoost
from hyperopt import hp, fmin, tpe, Trials

def catboost_objective(params):
    model = CatBoostClassifier(
        iterations=int(params['iterations']),
        learning_rate=params['learning_rate'],
        depth=int(params['depth']),
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=42,
        verbose=False
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return -scores.mean()  # Minimize negative F1

space = {
    'iterations': hp.choice('iterations', [500, 1000, 1500]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'depth': hp.choice('depth', [4, 6, 8, 10]),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
}

best = fmin(catboost_objective, space, algo=tpe.suggest, max_evals=100)
```

### Reducing False Positives

#### Contextual Filtering

```python
def apply_business_context_filter(prediction, user_metadata, content_metadata):
    # Reduce sensitivity for approved business processes
    if user_metadata.get('project_approval') and content_metadata.get('project_related'):
        prediction['risk_score'] *= 0.7
    
    # Account for user role and seniority
    if user_metadata.get('role') in ['executive', 'legal', 'hr']:
        prediction['risk_score'] *= 0.8
    
    # Time-based adjustments
    if is_deadline_period(user_metadata.get('department')):
        prediction['risk_score'] *= 0.9
    
    return prediction
```

#### Adaptive Thresholding

```python
class AdaptiveThresholdManager:
    def __init__(self):
        self.user_thresholds = {}
        self.global_threshold = 0.7
    
    def update_threshold(self, user_id, feedback_history):
        # Increase threshold for users with high false positive rates
        fp_rate = calculate_fp_rate(feedback_history)
        if fp_rate > 0.1:  # 10% false positive rate
            self.user_thresholds[user_id] = min(0.9, self.global_threshold + 0.1)
        elif fp_rate < 0.05:  # Very low false positive rate
            self.user_thresholds[user_id] = max(0.5, self.global_threshold - 0.1)
    
    def get_threshold(self, user_id):
        return self.user_thresholds.get(user_id, self.global_threshold)
```

### Trade-off Management

#### Performance vs. Accuracy Trade-offs

```python
# Configurable model complexity
class DLPModelConfig:
    FAST_MODE = {
        'bert_model': 'distilbert-base-uncased',
        'max_sequence_length': 256,
        'catboost_iterations': 500,
        'feature_selection_threshold': 0.01,
        'target_latency_ms': 200
    }
    
    BALANCED_MODE = {
        'bert_model': 'bert-base-uncased', 
        'max_sequence_length': 512,
        'catboost_iterations': 1000,
        'feature_selection_threshold': 0.005,
        'target_latency_ms': 500
    }
    
    ACCURACY_MODE = {
        'bert_model': 'bert-large-uncased',
        'max_sequence_length': 512,
        'catboost_iterations': 2000,
        'feature_selection_threshold': 0.001,
        'target_latency_ms': 1000
    }
```

#### Privacy vs. Detection Trade-offs

- **High Privacy**: Local processing, no content logging, differential privacy
- **Balanced**: Hashed content storage, role-based access, limited retention
- **High Detection**: Full content analysis, extended retention, detailed logging

---

## 🔹 Section 6: Common Pitfalls & Debugging Tips

### Training Phase Pitfalls

#### Data Quality Issues

**Symptom**: Model performs poorly on production data despite good validation scores **Common Causes:**

- Training data not representative of real user behavior
- Insufficient examples of sophisticated evasion techniques
- Imbalanced data with too few positive examples

**Solutions:**

```python
# Data quality validation
def validate_training_data(X, y):
    # Check class balance
    class_distribution = np.bincount(y) / len(y)
    if min(class_distribution) < 0.1:
        print("Warning: Severe class imbalance detected")
    
    # Check feature distributions
    for col in X.columns:
        if X[col].nunique() == 1:
            print(f"Warning: {col} has no variance")
        
        # Check for data leakage
        if col.endswith('_future') or 'target' in col.lower():
            print(f"Potential data leakage in {col}")
    
    return True
```

#### BERT Fine-tuning Issues

**Catastrophic Forgetting**: BERT loses general language understanding **Solution**: Use lower learning rates and gradual unfreezing

```python
# Gradual unfreezing strategy
def gradual_unfreeze_training(model, train_dataloader, num_epochs=3):
    # Freeze all layers initially
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Train classifier head only
    train_epoch(model, train_dataloader, optimizer, epoch=1)
    
    # Unfreeze last few layers
    for param in model.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True
    
    train_epoch(model, train_dataloader, optimizer, epoch=2)
    
    # Unfreeze all layers with lower learning rate
    for param in model.bert.parameters():
        param.requires_grad = True
    optimizer.param_groups[0]['lr'] *= 0.1
    
    train_epoch(model, train_dataloader, optimizer, epoch=3)
```

### Deployment Issues

#### Memory and Performance Problems

**Symptom**: Out of memory errors or excessive latency in production **Debugging Steps:**

```python
# Memory profiling
import psutil
import torch

def profile_model_memory():
    # Monitor GPU memory
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Monitor system memory
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.used / 1024**2:.2f} MB used / {memory.total / 1024**2:.2f} MB total")
    
    # Profile model components
    bert_memory = get_model_size(bert_model)
    catboost_memory = get_model_size(catboost_model)
    print(f"BERT model size: {bert_memory:.2f} MB")
    print(f"CatBoost model size: {catboost_memory:.2f} MB")
```

#### Feature Drift Detection

**Symptom**: Model performance degrades over time **Solution**: Implement comprehensive drift monitoring

```python
class DriftDetector:
    def __init__(self, reference_data):
        self.reference_stats = self.calculate_stats(reference_data)
    
    def detect_drift(self, current_data, threshold=0.05):
        drift_detected = {}
        
        for feature in current_data.columns:
            if feature in self.reference_stats:
                # Statistical test for numerical features
                if pd.api.types.is_numeric_dtype(current_data[feature]):
                    statistic, p_value = ks_2samp(
                        self.reference_stats[feature], 
                        current_data[feature]
                    )
                    drift_detected[feature] = p_value < threshold
                
                # Chi-square test for categorical features
                else:
                    ref_counts = self.reference_stats[feature].value_counts()
                    curr_counts = current_data[feature].value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    chi2, p_value = chisquare(curr_aligned, ref_aligned)
                    drift_detected[feature] = p_value < threshold
        
        return drift_detected
```

### Real-World Data Challenges

#### Handling Encrypted/Compressed Content

**Challenge**: Cannot analyze content that's encrypted or compressed **Solutions:**

```python
def handle_obfuscated_content(content, metadata):
    # Detect compression
    if is_compressed(content):
        try:
            # Attempt decompression for analysis
            decompressed = decompress_content(content)
            return analyze_content(decompressed)
        except:
            # Fall back to metadata analysis
            return analyze_metadata_only(metadata)
    
    # Detect encryption
    elif is_encrypted(content):
        # Use metadata and behavioral signals only
        return {
            'semantic_score': None,
            'behavioral_score': analyze_behavior(metadata),
            'risk_factors': ['encrypted_content', 'unusual_size']
        }
    
    return analyze_content(content)
```

#### Multi-Language Support

**Challenge**: BERT trained on English doesn't work well with other languages **Solution:**

```python
# Multi-language model selection
def select_bert_model(detected_language):
    language_models = {
        'en': 'bert-base-uncased',
        'es': 'bert-base-multilingual-cased',
        'zh': 'bert-base-chinese',
        'multilingual': 'bert-base-multilingual-cased'
    }
    
    return language_models.get(detected_language, 'bert-base-multilingual-cased')

# Language detection pipeline
def analyze_multilingual_content(content):
    detected_lang = detect_language(content)
    model = select_bert_model(detected_lang)
    
    return model.analyze(content)
```

### False Positive Debugging

#### Systematic FP Analysis

```python
def analyze_false_positives(predictions, actual_labels, feature_data):
    # Identify false positive cases
    fp_mask = (predictions == 1) & (actual_labels == 0)
    fp_cases = feature_data[fp_mask]
    
    # Analyze common patterns in false positives
    print("False Positive Analysis:")
    print(f"Total FPs: {fp_mask.sum()}")
    
    # Feature importance for FP cases
    if hasattr(model, 'feature_importances_'):
        fp_importance = analyze_feature_importance(fp_cases)
        print("Top features in false positives:")
        print(fp_importance.head(10))
    
    # User/content patterns
    fp_user_patterns = fp_cases.groupby('user_id').size().sort_values(ascending=False)
    print("Users with most false positives:")
    print(fp_user_patterns.head(10))
    
    return fp_cases
```

---

## 🔹 Section 7: Intern Implementation Checklist 🧠

### Pre-Development Setup

- [ ] **Environment Setup**: Install transformers, catboost, torch, pandas, scikit-learn
- [ ] **Data Collection**:
    - [ ] Gather sample emails, documents, chat logs
    - [ ] Collect behavioral data (timestamps, volumes, destinations)
    - [ ] Create synthetic sensitive content for testing
- [ ] **Baseline Implementation**: Simple keyword-based DLP for comparison
- [ ] **Evaluation Framework**: Define metrics (precision, recall, F1, latency)

### Week 1: Data Understanding & Preprocessing

- [ ] **Data Exploration**:
    - [ ] Analyze content types and formats
    - [ ] Understand user behavior patterns
    - [ ] Identify data quality issues
- [ ] **Preprocessing Pipeline**:
    - [ ] Text cleaning and normalization
    - [ ] BERT tokenization implementation
    - [ ] Behavioral feature extraction
- [ ] **Initial Labeling**: Create small labeled dataset for validation

### Week 2: BERT Implementation

- [ ] **Model Selection**: Choose appropriate BERT variant (base vs. distil)
- [ ] **Fine-tuning Setup**:
    - [ ] Prepare training data in correct format
    - [ ] Implement data loaders
    - [ ] Set up training loop with validation
- [ ] **Semantic Analysis**:
    - [ ] Implement content classification
    - [ ]