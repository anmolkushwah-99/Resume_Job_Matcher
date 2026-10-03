# Machine Learning Data Splitting Strategy: Preventing Data Leakage

## 1. The Core Problem: Row-Level vs. Candidate-Level Splitting

In a resume-job matching system, the training dataset consists of **pairs** $(Resume_i, Job_j)$. A single candidate resume $Resume_i$ is evaluated against multiple job postings $(Job_1, Job_2, \dots, Job_k)$.

### ⚠️ The Risk of Naive Row-Level Random Splitting
If rows of `raw_pairs.csv` are randomly shuffled into Train/Test splits:
* Row `(RES001, JOB001)` might end up in the **Training Set**.
* Row `(RES001, JOB005)` might end up in the **Test Set**.

This causes **severe data leakage**: the model memorizes candidate $RES001$'s specific phrasing, project vocabulary, and background during training, leading to artificially inflated test accuracy that fails when deployed on completely unseen candidates.

---

## 2. Grouped Candidate-Level Partitioning Strategy

To guarantee honest and generalizable model evaluation, partitioning **MUST be performed at the candidate/resume level** (`resume_id`):

```
                       All Resumes (25 Candidates)
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
  Train Resumes              Val Resumes                  Test Resumes
 (~70% of Candidates)      (~15% of Candidates)         (~15% of Candidates)
      │                            │                            │
      ▼                            ▼                            ▼
Train Pairs Dataset          Val Pairs Dataset            Test Pairs Dataset
(Only Train Candidates)      (Only Val Candidates)        (Only Test Candidates)
```

### Partitioning Rules:
1. **Disjoint Sets:** $\text{Candidates}_{\text{train}} \cap \text{Candidates}_{\text{val}} \cap \text{Candidates}_{\text{test}} = \emptyset$.
2. **Stratification by Domain:** Ensure diverse candidate categories (Backend, Frontend, ML, Data, Non-tech) are proportionately represented in Train, Validation, and Test sets using Stratified Group K-Fold techniques (`GroupKFold` or `StratifiedGroupKFold` from `scikit-learn`).
3. **Job Distribution:** Jobs may appear across splits, but candidates remain strictly quarantined to their designated split.

---

## 3. Recommended Split Ratios for Model Training

For future modeling phases (SVM, ANN, TF-IDF baseline):

| Partition | Candidate Percentage | Prototype Candidates | Purpose |
| :--- | :--- | :--- | :--- |
| **Train Set** | 70% | ~17–18 Resumes | Feature learning, weight optimization, SVM hyperplanes, ANN backpropagation |
| **Validation Set** | 15% | ~3–4 Resumes | Hyperparameter tuning, early stopping, regularization selection |
| **Test Set** | 15% | ~3–4 Resumes | Unbiased final evaluation of generalizability on unseen applicants |

---

## 4. Implementation in Python (Future Phase Reference)

When implementing the split in Phase 5/6/7, use `scikit-learn`:

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss.split(pairs_df, pairs_df['match_label'], groups=pairs_df['resume_id']))

train_pairs = pairs_df.iloc[train_idx]
temp_pairs = pairs_df.iloc[temp_idx]

# Split temp into validation and test sets
gss_val = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val.split(temp_pairs, temp_pairs['match_label'], groups=temp_pairs['resume_id']))

val_pairs = temp_pairs.iloc[val_idx]
test_pairs = temp_pairs.iloc[test_idx]
```

This guarantees 100% isolation across candidate resumes.
