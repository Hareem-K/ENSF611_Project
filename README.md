# Breast Cancer Classification - ENSF 611 Final Project
### Machine Learning for Software Engineers

By: Hareem Khan & Heena Heena

---

### Project Overview
This project develops a machine learning classification system to predict whether a breast tumour is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. Early detection is critical in medical decision-making, and ML-based decision-support tools can help identify concerning cases more reliably and efficiently.

Our goal was to compare multiple supervised learning models, evaluate their performance, and determine which model is most suitable for this dataset and context.

---

### Dataset Description
Dataset: Breast Cancer Wisconsin (Diagnostic)
Source:
- UCI Machine Learning Repository
- Also available through sklearn.datasets.load_breast_cancer

The dataset contains:
- 569 samples
- 30 numerical features extracted from digitized cell-nuclei images

Binary target:
- 0 = malignant
- 1 = benign

There were no missing values, but imputation logic was included to ensure robustness if applied to real-world data.

---

### Models Implemented
We evaluated three supervised classification models:
1. Logistic Regression
    - Scaled using StandardScaler
    - Hyperparameters tuned with GridSearchCV
2. Decision Tree Classifier
    - Used unscaled numerical features
    - Tuned over depth, split criteria, and leaf constraints
3. Support Vector Machine (SVM, RBF Kernel)
    - Required scaling
    - Grid search for optimal c and gamma

Each model followed the same workflow:
- Train/test split (80/20, stratified)
- 5-fold cross-validation
- hyperparameter tuning
- Evaluation using:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix
    - ROC Curve

---
### Key Results

#### Model Performance Summary
| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC    |
| ------------------- | -------- | --------- | ------ | -------- | ---------- |
| Logistic Regression | 0.9825   | 0.9861    | 0.9861 | 0.9861   | 0.9957     |
| Decision Tree       | 0.9211   | 0.9565    | 0.9167 | 0.9362   | 0.9163     |
| SVM (RBF)           | 0.9825   | 0.9861    | 0.9861 | 0.9861   | **0.9977** |

#### Combined ROC Curve
The notebook includes a combined ROC plot comparing all three models.
SVM’s curve consistently hugs the top-left corner, indicating superior overall separability.

#### Interpretation of Results
Both Logistic Regression and SVM (RBF) performed extremely well on this dataset. They achieved identical accuracy, precision, recall, and F1 scores.

The key deciding factor is ROC-AUC, which measures the model’s ability to distinguish malignant from benign tumours across all thresholds:
- SVM achieved the highest ROC-AUC (0.9977)
- Logistic Regression was slightly lower (0.9957)
- Decision Tree performed notably worse (0.9163)

While the difference between LR and SVM is small, in a medical context, even small improvements in false-negative reduction are meaningful. SVM’s ability to model non-linear boundaries makes it more robust in ambiguous cases.

**Final Recommendation**:
Support Vector Machine (RBF) is the best suited model for this dataset.

---
### Alignment with Proposal
We followed our original plan closely:
#### What stayed the same:
- Same dataset (Breast Cancer Wisconsin Diagnostic)
- Same models: Logistic Regression, Decision Tree, SVM
- Included scaling, cross-validation, and hyperparameter tuning
- Used ROC curve, confusion matrices, and evaluation metrics
- Performed model comparison & interpretation

#### Additional improvements (enhancements, not deviations):
- Added combined ROC curve for clearer visualization
- Included feature importance tables for LR and Decision Tree
- Added missing-value handling logic even though the dataset had none (to improve generalizability)

There were no deviations from the proposed workflow — only enhancements to improve clarity and interpretability.

---
### Repository Contents
- `ENSF611_Project.ipynb` — Full machine learning workflow, model training, tuning, evaluation, and visualizations  
- `README.md` — Project documentation and summary  
