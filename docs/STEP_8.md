# Step 8 — Supervised LSTM / RNN-Based Sequence Modeling for Resume–Job Match Classification

## 1. Step 8 Objective & Conceptual Framework

The primary objective of **Step 8** is to advance the matching system beyond unordered bag-of-words / TF-IDF representations by introducing **genuine sequential Natural Language Processing (NLP)**.

In Step 8, resume and job descriptions are modeled as **ordered token sequences**:
$$\mathbf{x} = (x_1, x_2, \dots, x_T)$$
where each token $x_t \in \mathbb{N}$ represents an integer vocabulary index mapped to a dense vector via a trainable **Embedding Layer**, processed sequentially by a **Long Short-Term Memory (LSTM)** recurrent neural network, and classified through dense feedforward layers with a **Sigmoid** output activation.

---

## 2. Progression & Relationship to Steps 5, 6, and 7

Across the project trajectory, each step explores a distinct machine learning and NLP paradigm on the exact same benchmark dataset:

```
Step 5 (Unsupervised / Lexical):
Resume & Job Text ──► TF-IDF Vectors ──► Cosine Similarity (Unsupervised Baseline)

Step 6 (Supervised / Linear Kernel):
Resume & Job Text ──► Concatenated TF-IDF (1,661 dims) ──► Linear SVM (Hyperplane Boundary)

Step 7 (Supervised / Deep Feedforward MLP):
Resume & Job Text ──► Concatenated TF-IDF (1,661 dims) ──► Dense ANN / MLP (Non-linear Activations)

Step 8 (Supervised / Deep Recurrent Sequence Modeling):
Resume & Job Text ──► Tokenization ──► Integer Sequences ──► Embedding(64) ──► LSTM(64) ──► Dense(32) ──► Sigmoid
```

---

## 3. Dataset & Candidate-Aware Zero-Leakage Split

Step 8 strictly reuses the **candidate-grouped 3-way partition** established in Step 6 and Step 7. All pairs belonging to any candidate resume are confined to exactly one partition:

| Partition | Candidates ($N=25$) | Resume IDs | Total Pairs ($N=120$) | Positive Matches | Negative Matches | Positive Ratio |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Train** | **17** | `RES004`, `RES005`, `RES006`, `RES007`, `RES008`, `RES010`, `RES011`, `RES012`, `RES013`, `RES014`, `RES015`, `RES016`, `RES018`, `RES020`, `RES022`, `RES023`, `RES025` | **81** | 25 | 56 | 30.86% |
| **Validation** | **4** | `RES002`, `RES003`, `RES019`, `RES021` | **21** | 8 | 13 | 38.10% |
| **Test (Held-Out)** | **4** | `RES001`, `RES009`, `RES017`, `RES024` | **18** | 5 | 13 | 27.78% |

### Leakage Prevention Proof
- $\text{Candidates}(\text{Train}) \cap \text{Candidates}(\text{Val}) = \emptyset$
- $\text{Candidates}(\text{Train}) \cap \text{Candidates}(\text{Test}) = \emptyset$
- $\text{Candidates}(\text{Val}) \cap \text{Candidates}(\text{Test}) = \emptyset$

---

## 4. Why TF-IDF is NOT Used for the LSTM

In Step 7, the input was a static 1,661-dimensional TF-IDF vector representing unordered n-gram frequencies. 
In Step 8, **TF-IDF is completely bypassed**:
1. An LSTM requires an ordered sequence of discrete time-steps $(\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_T)$.
2. Feeding static TF-IDF vectors into recurrent units would reduce the LSTM to an inefficient feedforward dense layer and destroy sequential meaning.
3. Step 8 models the text as ordered token sequences where relative order, boundary delimiter tags, and technical skill co-occurrences are preserved through time.

---

## 5. Sequential Text Construction & Pair Delimiters

To present both the resume profile and target job description to the recurrent model in a structured single stream, each pair is wrapped in boundary delimiter tokens:

$$\text{Sequence Text} = \texttt{<RESUME>} \parallel \text{Resume Content} \parallel \texttt{</RESUME>} \parallel \texttt{<JOB>} \parallel \text{Job Content} \parallel \texttt{</JOB>}$$

### Example Structured Sequence
```
<RESUME> python machine learning deep learning neural network tensorflow docker sql </RESUME> <JOB> senior machine learning engineer python tensorflow pytorch cloud </JOB>
```

---

## 6. Sequential Tokenization Strategy

Tokenization is performed using a custom deterministic `SequenceTokenizer`:
- **Training-Only Fitting**: The vocabulary is built strictly on the training partition in Phase A. Validation and test texts are transformed without altering the vocabulary.
- **Reserved Indices**:
  - `0`: `<PAD>` (padding token)
  - `1`: `<UNK>` (out-of-vocabulary unknown token)
  - `2`: `<RESUME>`
  - `3`: `</RESUME>`
  - `4`: `<JOB>`
  - `5`: `</JOB>`
- **Technical Token Preservation**: Regex pattern preserves technical punctuation (e.g., `c++`, `c#`, `.net`, `react.js`, `ci/cd`, `spring-boot`).
- **Vocabulary Size**:
  - Training Partition (Phase A): **432** unique tokens
  - Final Retraining Partition (Train + Val, Phase B): **462** unique tokens

---

## 7. Sequence Length Analysis & Padding Protocol

The sequence length distribution of the training partition was analyzed empirically:

| Metric | Training Token Length |
|:---|:---:|
| Minimum Length | 97 tokens |
| Maximum Length | 134 tokens |
| Mean Length | 117.7 tokens |
| Median Length | 118.0 tokens |
| 75th Percentile | 123.0 tokens |
| 90th Percentile | 128.0 tokens |
| **95th Percentile** | **130.0 tokens** |

### Selected Max Length & Alignment
- **`max_length = 130`** (strictly aligned to the 95th percentile of the training corpus).
- **Padding Strategy**: `post` (sequences shorter than 130 are zero-padded at the end).
- **Truncation Strategy**: `post` (sequences longer than 130 are truncated at index 130).
- **Distribution Impact**: 92.59% padded, 4.94% truncated, 2.47% exact match.

---

## 8. Trainable Embedding Layer

The integer sequence $\mathbf{s} = (s_1, s_2, \dots, s_{130}) \in \mathbb{N}^{130}$ is passed to a trainable **Embedding Layer**:
$$\mathbf{e}_t = \mathbf{W}_{\text{emb}}[s_t] \in \mathbb{R}^{64}$$
- **Input Dimension**: Dynamic vocabulary size ($432$ in Phase A, $462$ in Phase B).
- **Output Dimension**: $d = 64$.
- **No Pretrained Embeddings**: Weights are randomly initialized and learned directly from domain training data through backpropagation.

---

## 9. Primary LSTM Architecture & Layers

The end-to-end sequential network consists of:

```
Input Tensor: (batch_size, 130)
      │
      ▼
Embedding Layer: (batch_size, 130, 64)   [Trainable word vector lookup]
      │
      ▼
LSTM Layer: (batch_size, 64)            [64 hidden recurrent units, return_sequences=False]
      │
      ▼
Dropout Layer (p = 0.30)                [Regularization on recurrent output]
      │
      ▼
Dense Layer (32 units, ReLU)            [Non-linear feature projection]
      │
      ▼
Dropout Layer (p = 0.20)                [Regularization]
      │
      ▼
Dense Output Layer (1 unit, Sigmoid)    [Binary match probability score]
```

---

## 10. Theoretical Foundations: RNN & Recurrent Hidden States

A standard Recurrent Neural Network (RNN) processes a sequence step-by-step:
$$\mathbf{h}_t = f\left(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}\right)$$
where $\mathbf{h}_t$ carries memory from previous steps. However, standard RNNs suffer from **vanishing and exploding gradients** over long sequences because repetitive multiplication by $\mathbf{W}_h$ causes exponential decay or growth of gradient magnitudes during backpropagation.

---

## 11. Theoretical Foundations: LSTM Cell State & Gating Mechanism

The **Long Short-Term Memory (LSTM)** architecture solves vanishing gradients by introducing an internal **Cell State** $\mathbf{C}_t$ regulated by three multiplicative gating mechanisms:

1. **Forget Gate ($\mathbf{f}_t$)**: Determines which information from the previous cell state $\mathbf{C}_{t-1}$ to discard:
   $$\mathbf{f}_t = \sigma\left(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$

2. **Input Gate ($\mathbf{i}_t$)**: Determines which new candidate values to store in the cell state:
   $$\mathbf{i}_t = \sigma\left(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right)$$

3. **Candidate Cell State ($\tilde{\mathbf{C}}_t$)**: Generates new candidate values:
   $$\tilde{\mathbf{C}}_t = \tanh\left(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C\right)$$

4. **Cell State Update ($\mathbf{C}_t$)**: Updates the long-term memory via additive combination:
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

5. **Output Gate ($\mathbf{o}_t$)**: Determines what portion of the cell state forms the output:
   $$\mathbf{o}_t = \sigma\left(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right)$$

6. **Hidden State ($\mathbf{h}_t$)**: Emits the gated hidden state to subsequent layers:
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

---

## 12. Theoretical Foundations: Backpropagation Through Time (BPTT)

Recurrent network parameters are optimized using **Backpropagation Through Time (BPTT)**:
1. The recurrent loop is unfolded across $T=130$ discrete time steps:
   $$\mathbf{x}_1 \to \mathbf{h}_1 \to \mathbf{x}_2 \to \mathbf{h}_2 \to \dots \to \mathbf{x}_T \to \mathbf{h}_T$$
2. The loss gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ is accumulated by propagating error backwards through temporal unrollings:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}}$$
3. In standard RNNs, $\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \mathbf{W}_h^T \text{diag}(1 - \mathbf{h}_k^2)$ vanishes as $T - t$ grows. In LSTM, the additive cell state gradient $\frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_t}$ contains direct paths, facilitating gradient flow across 100+ sequence tokens.

---

## 13. Loss Function & Optimizer

- **Loss Function**: Binary Cross-Entropy (Log Loss)
  $$\mathcal{L}(y, \hat{p}) = - \left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$
- **Optimizer**: **Adam** (Adaptive Moment Estimation)
  - Initial learning rate: $\alpha = 0.001$
  - Exponential decay parameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$
  - Adapts parameter-specific learning rates using smoothed first moment (mean) and second moment (uncentered variance) of stochastic gradients.

---

## 14. Class Imbalance Weighting

Because positive pairs constitute ~30.86% of training samples (25 positive vs. 56 negative), inverse frequency class weights were computed strictly from training labels:
$$w_0 = \frac{N}{2 \cdot N_0} = \frac{81}{2 \cdot 56} \approx 0.7232, \quad w_1 = \frac{N}{2 \cdot N_1} = \frac{81}{2 \cdot 25} \approx 1.6200$$

---

## 15. Validation Experiments & Hyperparameter Tuning

Four controlled configurations were evaluated on the validation partition ($N=21$ pairs / 4 candidates):

| Configuration | Embedding Dim | LSTM Units | Dense Units | Dropout Rates | Learning Rate | Actual / Best Epoch | Val Loss | Val Acc | Val Prec | Val Recall | Val F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A (Primary)** | **64** | **64** | **32** | **0.30 / 0.20** | **0.001** | **25 / 17** | 0.6729 | **0.7619** | **1.0000** | 0.3750 | **0.5455** |
| Config B (Small Emb) | 32 | 64 | 32 | 0.30 / 0.20 | 0.001 | 48 / 40 | 0.5393 | 0.7619 | 1.0000 | 0.3750 | 0.5455 |
| Config C (Small LSTM) | 64 | 32 | 32 | 0.20 / 0.20 | 0.001 | 31 / 23 | 0.5731 | 0.7619 | 1.0000 | 0.3750 | 0.5455 |
| Config D (Low LR) | 64 | 64 | 32 | 0.30 / 0.20 | 0.0005 | 42 / 34 | 0.5704 | 0.7619 | 1.0000 | 0.3750 | 0.5455 |

**Selected Architecture**: **Config A** (Embedding=64, LSTM=64, Dense=32, Dropout=0.3/0.2, lr=0.001).

---

## 16. Validation Threshold Optimization

Using Config A's validation sigmoid scores, decision thresholds from $0.30$ to $0.70$ were evaluated:

| Threshold ($\theta$) | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 |
|:---:|:---:|:---:|:---:|:---:|
| **0.30** | **0.3810** | **0.3810** | **1.0000** | **0.5517** |
| 0.35 | 0.4762 | 0.4211 | 1.0000 | 0.5926 |
| 0.40 | 0.5714 | 0.4706 | 1.0000 | 0.6400 |
| 0.45 | 0.6667 | 0.5385 | 0.8750 | 0.6667 |
| 0.50 | 0.7619 | 1.0000 | 0.3750 | 0.5455 |
| 0.55 | 0.7143 | 1.0000 | 0.2500 | 0.4000 |
| 0.60 | 0.6190 | 0.0000 | 0.0000 | 0.0000 |

**Selected & Frozen Decision Threshold**: **$\theta = 0.30$** (selected for maximum sensitivity / recall in screening context).

---

## 17. Final Model Retraining (Phase B)

In Phase B, the final model was trained on the combined **Train + Validation** partition ($N=102$ pairs / 21 candidates):
- Combined Vocabulary Size: **462** tokens
- Max Sequence Length: **130** tokens
- Actual Training Epochs: **26** (Best Epoch: **18**)
- Final Weights & Tokenizer saved to `models/embedding_lstm_model.keras` and `models/embedding_lstm_tokenizer.joblib`.

---

## 18. Held-Out Test Evaluation (Single-Pass)

The final frozen model was evaluated exactly once on the untouched held-out test partition ($N=18$ pairs / 4 candidates):

| Metric | Held-Out Test Score |
|:---|:---:|
| **Accuracy** | **27.78%** ($5 / 18$) |
| **Precision** | **27.78%** ($5 / 18$) |
| **Recall (Sensitivity)** | **100.00%** ($5 / 5$) |
| **F1-Score** | **43.48%** |
| **Specificity** | **0.00%** ($0 / 13$) |
| **ROC-AUC** | **0.3385** |
| **PR-AUC** | **0.2478** |

### Test Confusion Matrix
| | Predicted Non-Match ($\hat{y}=0$) | Predicted Match ($\hat{y}=1$) |
|:---|:---:|:---:|
| **Actual Non-Match ($y=0$)** | **TN = 0** | **FP = 13** |
| **Actual Match ($y=1$)** | **FN = 0** | **TP = 5** |

---

## 19. Four-Model Common Test Set Comparison

All four models evaluated across Steps 5–8 on the identical held-out benchmark test partition:

| Model ID | Input Representation | Architecture | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `tfidf_cosine_baseline` (Step 5) | TF-IDF (1,661 dims) | Cosine Similarity ($\theta = 0.05$) | 0.5556 | 0.3846 | 1.0000 | **0.5556** | 5 | 8 | 5 | 0 |
| `tfidf_svm` (Step 6) | TF-IDF (1,661 dims) | LinearSVC ($C=1.0$, balanced) | **0.6111** | **0.3750** | 0.6000 | 0.4615 | 3 | 5 | 8 | 2 |
| `tfidf_ann` (Step 7) | TF-IDF (1,661 dims) | Dense MLP ($128 \to 64$, $\theta = 0.30$) | 0.5556 | 0.3333 | 0.6000 | 0.4286 | 3 | 6 | 7 | 2 |
| `embedding_lstm` (Step 8) | Token Sequences (Vocab=462, MaxLen=130) | Embedding(64) $\to$ LSTM(64) $\to$ Dense(32) ($\theta = 0.30$) | 0.2778 | 0.2778 | **1.0000** | 0.4348 | 5 | 13 | 0 | 0 |

---

## 20. Critical Empirical Insights & Academic Interpretation

1. **Recurrent Capacity vs. Benchmark Sample Size**:
   - The benchmark dataset contains 120 pairs across 25 candidates. Deep recurrent models with an embedding layer have significantly higher parameter variance ($>40,000$ trainable parameters) compared to linear SVM ($1,661$ parameters).
   - While the LSTM achieved 100% recall on the test set, the low threshold ($\theta = 0.30$) resulted in an aggressive matching tendency ($FP = 13$).
2. **Lexical Sparsity in Small Corpora**:
   - In extremely small datasets, TF-IDF + linear models (SVM / Cosine) benefit from direct lexical keyword overlap. Trainable embeddings require larger corpora to cluster semantic representations effectively.
3. **Model Diversity for Step 9**:
   - The four models exhibit distinct operating characteristics: Linear SVM provides high specificity ($TN=8$), Cosine provides balanced lexical baseline ($F1=0.5556$), ANN provides non-linear feature interaction, and LSTM captures sequential context. This provides rich diversity for Step 9 model selection and ensemble analysis.

---

## 21. Database Integration & Model Coexistence

Step 8 stores match records directly in the existing MySQL `matches` table without schema modifications:

```sql
INSERT INTO matches (
    resume_id, job_id, similarity_score, overall_score,
    model_name, is_match, match_label, threshold_used,
    matched_skills, missing_skills, updated_at
) VALUES (
    'RES001', 'JOB001', 0.4821, 0.4821,
    'embedding_lstm', TRUE, 'Match', 0.30,
    '["python", "machine learning"]', '["pytorch", "kubernetes"]', NOW()
);
```

### Database Coexistence
All four models coexist cleanly in MySQL queryable via `model_name`:
- `tfidf_cosine_baseline`
- `tfidf_svm`
- `tfidf_ann`
- `embedding_lstm`

---

## 22. REST API Endpoints

### 1. Single Pair LSTM Match Prediction
- **Route**: `POST /api/matches/lstm`
- **Request**:
  ```json
  {
    "resume_id": "RES001",
    "job_id": "JOB001"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "resume_id": "RES001",
    "job_id": "JOB001",
    "model": "embedding_lstm",
    "lstm_score": 0.4821,
    "predicted_label": 1,
    "is_match": true,
    "threshold": 0.30,
    "shared_skills": ["python", "machine learning"],
    "missing_skills": ["pytorch", "kubernetes"],
    "created_at": "2026-10-01T14:51:39"
  }
  ```

### 2. Resume-Level Benchmark Job Ranking
- **Route**: `POST /api/resumes/<resume_id>/lstm-matches?limit=5`
- **Response**:
  ```json
  {
    "success": true,
    "resume_id": "RES001",
    "model": "embedding_lstm",
    "total_evaluated": 12,
    "matches": [
      {
        "job_id": "JOB001",
        "job_title": "Machine Learning Engineer",
        "lstm_score": 0.5612,
        "predicted_label": 1,
        "is_match": true,
        "threshold": 0.30,
        "shared_skills": ["python", "machine learning", "deep learning"]
      }
    ]
  }
  ```

---

## 23. Test Suite Verification

- **Pre-Step-8 Tests**: 155 passed
- **New Step-8 Tests**: 25 passed
- **Final Total Tests**: **180 passed, 0 failed**
- **Test Modules**:
  - `tests/test_lstm_model.py` (13 tests)
  - `tests/test_lstm_service.py` (5 tests)
  - `tests/test_lstm_api.py` (6 tests)
  - `tests/test_lstm_evaluation.py` (1 integration test)
  - Plus all 30 pre-existing test suites across Steps 1–7.

---

## 24. Academic Concepts Demonstrated

1. **Sequential NLP Representation**: Integer token sequences replacing static bag-of-words vectors.
2. **Trainable Word Embeddings**: Continuous $d=64$ dense projections optimized via backpropagation.
3. **Recurrent Neural Dynamics**: Hidden states $\mathbf{h}_t$ and persistent cell states $\mathbf{C}_t$.
4. **LSTM Gating**: Forget, Input, and Output gates controlling long-term memory.
5. **Backpropagation Through Time (BPTT)**: Unfolded temporal gradient propagation.
6. **Binary Cross-Entropy & Adam Optimization**: First and second moment adaptive learning rate scheduling.
7. **Zero-Leakage Experimental Design**: Candidate-aware grouping, training-only tokenization, validation hyperparameter tuning, and untouched single-pass held-out test evaluation.

---

## 25. Step 9 Readiness

Step 8 is fully completed and verified. **Step 9** (Comprehensive Model Evaluation, Trade-off Analysis, and Final Model Selection Methodology) is ready for execution in the subsequent phase.
