# Step 7 — Supervised ANN / MLP Classification for Resume-Job Matching

**AI-Based Resume Screening and Job Matching System**  
**Phase 7: Artificial Neural Networks, Backpropagation & Supervised Matching**  
**Status: COMPLETED**

---

## 1. Executive Summary & Objective

Step 7 introduces the project's **first Deep Learning / Multi-Layer Perceptron (ANN/MLP)** model for binary resume-job match classification. Built with **TensorFlow / Keras**, the neural network maps high-dimensional TF-IDF textual feature representations into non-linear latent spaces to predict relevance between candidate resumes and benchmark job descriptions (`match_label`).

### Key Deliverables Completed:
1. **Multi-Layer Perceptron Architecture** ([`src/models/ann_classifier.py`](file:///d:/advance%20ai/Resume_Job_Matcher/src/models/ann_classifier.py)): Compact Keras Sequential model featuring dynamic input dimensions, Dense layers with Rectified Linear Unit (ReLU) activations, Dropout regularization, and a Sigmoid output unit.
2. **Backpropagation & Adam Optimization**: Optimized via Binary Cross-Entropy loss, mini-batch gradient descent (batch size = 16), dynamic learning rate ($10^{-3}$), and class weighting for positive class compensation.
3. **Candidate-Aware Leak-Free Evaluation**: Reused the exact candidate-aware 3-way partition from Step 6 ([`data/processed/evaluation/ann_dataset_split.json`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_dataset_split.json)) guaranteeing **zero candidate overlap** across Train (17 candidates / 81 pairs), Validation (4 candidates / 21 pairs), and Test (4 candidates / 18 pairs).
4. **Validation-Based Hyperparameter Exploration & Threshold Optimization**: Explored 4 distinct network configurations and swept classification thresholds $\theta \in [0.30, 0.70]$ strictly on validation data to prevent test contamination.
5. **Two-Phase Training Protocol**: Phase A explored architectures on Train/Val; Phase B retrained the finalized configuration on Train+Val (102 pairs / 1,661 features) before evaluating once on the held-out test partition.
6. **Model Artifact Persistence**: Serialized native Keras model (`tfidf_ann_model.keras`), fitted vectorizer (`tfidf_ann_vectorizer.joblib`), and training metadata (`tfidf_ann_metadata.json`) in [`models/`](file:///d:/advance%20ai/Resume_Job_Matcher/models/).
7. **Multi-Model MySQL Coexistence** ([`src/models/ann_service.py`](file:///d:/advance%20ai/Resume_Job_Matcher/src/models/ann_service.py)): Persists ANN sigmoid scores into `matches` under `model_name="tfidf_ann"` alongside Step 5 (`tfidf_cosine_baseline`) and Step 6 (`tfidf_svm`) records.
8. **Privacy-Safe REST API Endpoints** ([`app/routes.py`](file:///d:/advance%20ai/Resume_Job_Matcher/app/routes.py)): `POST /api/matches/ann` and `POST /api/resumes/<resume_id>/ann-matches`.
9. **Head-to-Head 3-Model Comparison**: Evaluated Step 5, Step 6, and Step 7 on the exact same held-out test partition.
10. **100% Automated Test Coverage**: 155 / 155 passing tests across 23 test modules.

---

## 2. Theoretical Foundations & Mathematical Formulation

### 2.1 Why Artificial Neural Networks?
While Step 5 computed linear angular distance and Step 6 learned a linear separating hyperplane ($\mathbf{w} \cdot \mathbf{x} + b = 0$), text relationships often exhibit **non-linear feature interactions**. Multi-Layer Perceptrons project TF-IDF feature vectors into dense hidden representations where complex feature conjuncts can be captured.

### 2.2 Forward Propagation in Dense Layers
For an input vector $\mathbf{x} \in \mathbb{R}^{V}$ (where $V$ is the TF-IDF vocabulary dimension):

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}$$

$$\mathbf{a}^{[1]} = \text{ReLU}\left(\mathbf{z}^{[1]}\right) = \max\left(0, \mathbf{z}^{[1]}\right)$$

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

$$\mathbf{a}^{[2]} = \text{ReLU}\left(\mathbf{z}^{[2]}\right) = \max\left(0, \mathbf{z}^{[2]}\right)$$

$$\hat{y} = \sigma\left(\mathbf{z}^{[3]}\right) = \frac{1}{1 + e^{-\mathbf{z}^{[3]}}}$$

Where:
- $\mathbf{W}^{[1]} \in \mathbb{R}^{128 \times V}, \mathbf{b}^{[1]} \in \mathbb{R}^{128}$: Layer 1 weights and biases.
- $\mathbf{W}^{[2]} \in \mathbb{R}^{64 \times 128}, \mathbf{b}^{[2]} \in \mathbb{R}^{64}$: Layer 2 weights and biases.
- $\mathbf{W}^{[3]} \in \mathbb{R}^{1 \times 64}, \mathbf{b}^{[3]} \in \mathbb{R}$: Output layer weights and bias.
- $\hat{y} \in [0, 1]$: Continuous model sigmoid score representing the activation for Class 1 (Match).

### 2.3 Loss Function: Binary Cross-Entropy
Given ground-truth binary label $y \in \{0, 1\}$ and prediction $\hat{y}$:

$$\mathcal{L}(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

For a mini-batch of $M$ samples with class weights $w_{y_i}$:

$$J(\mathbf{W}, \mathbf{b}) = - \frac{1}{M} \sum_{i=1}^{M} w_{y_i} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 2.4 Backpropagation & Adam Optimization
1. **Gradient Computation**: Error terms $\boldsymbol{\delta}^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$ are propagated backward through layers using the chain rule:
   $$\boldsymbol{\delta}^{[3]} = \hat{y} - y$$
   $$\boldsymbol{\delta}^{[2]} = \left( {\mathbf{W}^{[3]}}^T \boldsymbol{\delta}^{[3]} \right) \odot \text{ReLU}'\left(\mathbf{z}^{[2]}\right)$$
   $$\boldsymbol{\delta}^{[1]} = \left( {\mathbf{W}^{[2]}}^T \boldsymbol{\delta}^{[2]} \right) \odot \text{ReLU}'\left(\mathbf{z}^{[1]}\right)$$
2. **Adam Optimizer**: Tracks first moment $m_t$ (mean) and second moment $v_t$ (uncentered variance) of gradients to adapt learning rates per parameter:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   Where $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$.

---

## 3. Candidate-Aware Dataset Splitting (Leak-Free Design)

To prevent candidate information leakage across partitions, all pairs involving the same resume/candidate are placed into exactly one partition.

```
Total: 25 Candidates / 120 Pairs
├── Train: 17 Candidates (68.0%) / 81 Pairs (25 Pos, 56 Neg)
├── Validation: 4 Candidates (16.0%) / 21 Pairs (8 Pos, 13 Neg)
└── Test: 4 Candidates (16.0%) / 18 Pairs (5 Pos, 13 Neg)
```

### Partition Overlap Verification
- $\text{Train} \cap \text{Validation} = \emptyset$ (0 candidates)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 candidates)
- $\text{Validation} \cap \text{Test} = \emptyset$ (0 candidates)

Stored as a machine-readable JSON artifact in [`data/processed/evaluation/ann_dataset_split.json`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_dataset_split.json).

---

## 4. Hyperparameter Exploration on Validation Partition

During Phase A, TF-IDF vectorization was fitted **strictly on the 81 Training pairs** ($V = 1,493$ features). 4 candidate network configurations were trained on Train and evaluated on the untouched Validation partition (21 pairs):

| Configuration | Hidden Layers | Dropout | Learning Rate | Batch Size | Actual Epochs | Best Epoch | Val Loss | Val Acc | Val Prec | Val Recall | Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A (Selected)** | **(128, 64)** | **(0.3, 0.2)** | **0.001** | **16** | **16** | **8** | **0.6446** | **0.7143** | **0.7500** | **0.3750** | **0.5000** |
| Config B | (64, 32) | (0.2, 0.2) | 0.001 | 16 | 22 | 14 | 0.6492 | 0.7143 | 0.7500 | 0.3750 | 0.5000 |
| Config C | (128, 64) | (0.3, 0.2) | 0.0005 | 16 | 36 | 28 | 0.6411 | 0.6667 | 0.6000 | 0.3750 | 0.4615 |
| Config D (No Drop) | (128, 64) | (0.0, 0.0) | 0.001 | 16 | 17 | 9 | 0.6441 | 0.7143 | 0.7500 | 0.3750 | 0.5000 |

*Saved to [`data/processed/evaluation/ann_hyperparameter_results.csv`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_hyperparameter_results.csv).*

---

## 5. Validation Threshold Optimization

Sweeping classification threshold $\theta \in [0.30, 0.70]$ on the validation set using Config A:

| Threshold ($\theta$) | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 |
| :--- | :--- | :--- | :--- | :--- |
| **0.30 (Selected)** | **0.4286** | **0.4000** | **1.0000** | **0.5714** |
| 0.35 | 0.4762 | 0.3846 | 0.6250 | 0.4762 |
| 0.40 | 0.5238 | 0.4000 | 0.5000 | 0.4444 |
| 0.45 | 0.5714 | 0.4286 | 0.3750 | 0.4000 |
| 0.50 | 0.7143 | 0.7500 | 0.3750 | 0.5000 |
| 0.55 | 0.7143 | 1.0000 | 0.2500 | 0.4000 |
| 0.60 | 0.7143 | 1.0000 | 0.2500 | 0.4000 |
| 0.65 | 0.7143 | 1.0000 | 0.2500 | 0.4000 |
| 0.70 | 0.6190 | 0.0000 | 0.0000 | 0.0000 |

*Threshold $\theta^* = 0.30$ maximized validation F1 ($0.5714$) and was frozen for test evaluation. Saved to [`data/processed/evaluation/ann_threshold_results.csv`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_threshold_results.csv).*

---

## 6. Phase B: Final Model Retraining & Held-Out Test Evaluation

The finalized architecture (Config A, $\theta^* = 0.30$) was retrained on combined **Train + Validation data** (102 pairs / 21 candidates), fitting a fresh vocabulary of **1,661 TF-IDF feature dimensions**.

The retrained model was evaluated **exactly once** on the untouched held-out test partition (18 pairs / 4 candidates):

### Final Test Metrics:
- **Accuracy**: $55.56\%$ ($10 / 18$)
- **Precision**: $33.33\%$
- **Recall**: $60.00\%$ ($3 / 5$ relevant matches identified)
- **F1-Score**: $0.4286$
- **Specificity**: $53.85\%$ ($7 / 13$ non-matches rejected)
- **ROC-AUC**: $0.5538$
- **PR-AUC**: $0.3958$

### Test Confusion Matrix:
```
                      Predicted Non-Match (0)    Predicted Match (1)
Actual Non-Match (0)           7 (TN)                    6 (FP)
Actual Match (1)               2 (FN)                    3 (TP)
```

*Saved to [`data/processed/evaluation/ann_metrics.json`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_metrics.json), [`ann_classification_report.txt`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_classification_report.txt), [`ann_confusion_matrix.png`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_confusion_matrix.png), and [`ann_training_curves.png`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_training_curves.png).*

---

## 7. Head-to-Head Model Comparison (Steps 5, 6, and 7)

All three models evaluated on the **exact same held-out test partition** (4 candidates, 18 pairs):

| Model | Model Architecture & Protocol | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 5: tfidf_cosine_baseline** | Cosine Similarity Baseline ($\theta = 0.05$) | 0.5556 | 0.3846 | **1.0000** | **0.5556** | 5 | 8 | 5 | 0 |
| **Step 6: tfidf_svm** | Linear Support Vector Machine ($C=1.0$, balanced) | **0.6111** | **0.3750** | 0.6000 | 0.4615 | 3 | 5 | **8** | 2 |
| **Step 7: tfidf_ann** | Multi-Layer Perceptron ($128 \to 64$, $\theta=0.30$) | 0.5556 | 0.3333 | 0.6000 | 0.4286 | 3 | 6 | 7 | 2 |

### Empirical Insights:
1. **SVM achieves highest Accuracy and Specificity**: The linear maximum-margin formulation effectively penalizes spurious lexical matches, achieving the highest test accuracy ($61.11\%$) and lowest false positive count ($5$ FP vs $8$ FP in baseline).
2. **ANN captures non-linear features with competitive Recall**: ANN matches SVM recall ($60.00\%$) and rejects over half of non-matches ($7$ TN), but exhibits slightly higher variance due to parameter density on a small training corpus ($N=102$).
3. **Cosine Baseline exhibits high sensitivity**: Achieving $100\%$ recall but low precision due to over-triggering on shared stopword/domain vocabulary.

*Saved to [`data/processed/evaluation/ann_model_comparison.csv`](file:///d:/advance%20ai/Resume_Job_Matcher/data/processed/evaluation/ann_model_comparison.csv).*

---

## 8. Database Integration & Multi-Model Coexistence

Predictions are stored transactionally in MySQL table `matches` without schema modifications:

```sql
SELECT model_name, overall_score, similarity_score FROM matches WHERE resume_id = 'RES001' AND job_id = 'JOB001';
```

| model_name | overall_score | similarity_score | Description |
| :--- | :--- | :--- | :--- |
| `tfidf_cosine_baseline` | 0.2845 | 0.2845 | Step 5 TF-IDF Cosine Similarity |
| `tfidf_svm` | 0.8241 | NULL | Step 6 Linear SVM Decision Score |
| `tfidf_ann` | 0.6312 | NULL | Step 7 ANN Sigmoid Match Score |

---

## 9. Privacy-Safe API Endpoints

### 9.1 Single-Pair ANN Match (`POST /api/matches/ann`)
**Request**:
```json
{
  "resume_id": "RES001",
  "job_id": "JOB001"
}
```
**Response (200 OK)**:
```json
{
  "success": true,
  "resume_id": "RES001",
  "job_id": "JOB001",
  "model": "tfidf_ann",
  "predicted_label": 1,
  "is_match": true,
  "ann_score": 0.6312,
  "threshold": 0.30,
  "shared_skills": ["Python", "Machine Learning", "SQL"]
}
```

### 9.2 Candidate Resume Against All Jobs (`POST /api/resumes/<resume_id>/ann-matches`)
Evaluates the candidate resume against all benchmark jobs and returns ranked matches sorted descending by `ann_score`.

---

## 10. Academic Limitations & Ethics

1. **Small Sample Constraint**: The dataset contains 25 candidates and 120 pairs. Deep learning models generally perform best with thousands of samples; results here illustrate learning dynamics rather than population generalizability.
2. **Lexical Representation**: TF-IDF vectors do not preserve sequential word ordering or semantic embeddings (e.g. Word2Vec/BERT).
3. **No Autonomous Hiring Decisions**: The ANN score is an academic decision-support activation value, not an employment guarantee.

---

## 11. Step 8 Readiness

With Step 5 (Cosine Similarity), Step 6 (Linear SVM), and Step 7 (Multi-Layer Perceptron) fully validated, the project is ready for **Step 8: LSTM / RNN-Based Sequence Modeling for Resume-Job Matching**.
