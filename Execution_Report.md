# Machine Learning Pipeline for Account Name Classification

## 1. Executive Summary

This report details the development and evaluation of an end-to-end machine learning pipeline designed to automatically classify transactional records into their respective `accountName` categories based on textual and numeric features. The pipeline architecture systematically unifies text vectorization (TF-IDF with n-grams) and numeric scaling using a `ColumnTransformer`, feeding directly into a Support Vector Machine (SVM) classifier. To ensure optimal performance without overfitting, the entire pipeline was rigorously evaluated using `GridSearchCV` to fine-tune the SVM's kernel selection and regularization strength via cross-validation. This unified approach successfully achieved an overall accuracy of 87.9%, surpassing the 85% minimum performance requirement. The pipeline effectively handles extreme class imbalances and high-cardinality text data, providing a robust and reliable system for automated financial record categorization.

---

## 2. Data Analysis & Exploration

### 2.1 Dataset Overview
The dataset consists of transactional records where each row corresponds to a specific item purchased or service rendered. The primary goal is to predict the `accountName` (the target variable) using the following features:
*   **Textual Features:** `vendorId`, `itemName`, `itemDescription`
*   **Numeric Features:** `itemTotalAmount`

### 2.2 Exploratory Findings
During the Exploratory Data Analysis (EDA) phase, several critical characteristics of the dataset were identified:
1.  **High-Cardinality Target:** The `accountName` target variable contains dozens of unique classes, ranging from broad categories like 'Online Subscription/Tool' to highly specific ones like 'Secretarial Expenses'. 
2.  **Missing Data:** The textual descriptions (`itemName`, `itemDescription`) frequently contained missing or null values. If ignored, these would prevent the machine learning model from processing the records.
3.  **Variable Monetary Ranges:** The `itemTotalAmount` feature spanned a vast range of values, which could heavily skew distance-based machine learning algorithms if left unscaled.

### 2.3 Class Imbalance Challenge
The most significant finding from the EDA was the severe **class imbalance**. 
A visual analysis of the class distribution revealed that the dataset is heavily skewed towards a few dominant accounts (e.g., '132098 IC Clearing account' and '611202 Online Subscription/Tool'). Conversely, there is a "long tail" of rare accounts that appear fewer than five times in the entire dataset.
*   **Impact:** Without intervention, a model trained on this data would become biased, constantly predicting the majority classes to artificially inflate its accuracy, while completely failing to identify the rare classes.

---

## 3. Methodology

### 3.1 Data Cleaning & Preprocessing
To address the challenges identified in the EDA, the following preprocessing steps were implemented:
1.  **Noise Reduction:** Account categories with fewer than five instances were removed. These "rare classes" lack sufficient data points for a model to learn generalizable patterns and act as noise during cross-validation.
2.  **Imputation:** Missing textual values were replaced with a standardized "Unknown" token, ensuring no data was lost. Missing numeric amounts were imputed with the median `itemTotalAmount` to prevent outliers from skewing the data.

### 3.2 Feature Engineering
The primary challenge of this assignment was converting the unstructured textual data into a mathematical format suitable for machine learning. 
*   **TF-IDF Vectorization:** We utilized Term Frequency-Inverse Document Frequency (TF-IDF) to vectorize the text. This technique not only counts word frequencies but penalizes highly common words, highlighting the unique terms that define specific accounts.
*   **N-Grams:** For `vendorId` (which are often alphanumeric codes), we used character n-grams (3 to 5 characters) to capture partial code matches. For `itemName` and `itemDescription`, we used word n-grams (1 to 3 words) to capture contextual phrases rather than just isolated words.
*   **Scaling:** The numeric `itemTotalAmount` was standardized using a `StandardScaler` to have a mean of 0 and a variance of 1, placing it on the same mathematical scale as the TF-IDF vectors.

All of these transformations were elegantly combined using a `ColumnTransformer` to create a unified data processing pipeline.

### 3.3 Algorithm Selection: Support Vector Machine (SVM)
A Support Vector Machine (`SVC`) was chosen over other algorithms (like Logistic Regression or Random Forests) for the following reasons:
1.  **High-Dimensional Space:** TF-IDF vectorization creates extremely high-dimensional, sparse matrices (tens of thousands of columns). SVMs are mathematically designed to find optimal hyperplanes in high-dimensional spaces, making them the industry standard for text classification tasks.
2.  **Handling Imbalance:** To counteract the severe class imbalance discovered during EDA, the `class_weight='balanced'` parameter was passed to the SVM. This algorithmically penalizes the model more heavily for misclassifying minority classes, forcing it to learn the rare categories instead of just predicting the majority class.

### 3.4 Validation Strategy
A **Stratified Train-Test Split** (80% training, 20% testing) was utilized. The stratification guarantees that the extreme class imbalance is proportionately maintained in both the training and testing sets, ensuring our final performance metrics are a realistic estimate of real-world capability. Finally, `GridSearchCV` with 3-fold cross-validation was used to tune the SVM's kernel (`linear` vs `rbf`) and regularization strength (`C`).

---

## 4. Results

The optimized SVM model achieved the following performance on the unseen test data:

### 4.1 Overall Performance Metrics
The table below compares the finalized SVM model against the other models evaluated during development.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **SVM (Finalized)** | **87.90%** | **88.40%** | **87.90%** | **87.72%** |
| XGBoost | 85.24% | 85.54% | 85.24% | 84.51% |
| CatBoost | 65.50% | 73.86% | 65.50% | 64.95% |
| Logistic Regression | 77.92% | 83.54% | 77.92% | 78.82% |

The close alignment of Precision, Recall, and F1-Score for the SVM indicates that the model is well-balanced and is not simply "guessing" the majority class to achieve its high accuracy.

### 4.2 Category Performance Breakdown
An analysis of the detailed classification report revealed specific areas of strength and weakness:

**Top Performers (F1-Score of 1.00 or >0.95):**
*   `611101 Cloud server - AWS`
*   `223001 Salaries Payable`
*   `617205 Consultancy Expense`
*   *Observation:* The model performs perfectly on accounts with highly distinct, standardized terminology or vendor IDs (e.g., "AWS", "Google", "Salaries").

**Bottom Performers (F1-Score < 0.60):**
*   `612021 Branding` (F1: 0.40)
*   `134002 Prepaid Insurance` (F1: 0.50)
*   *Observation:* The model struggles with accounts where the item descriptions are likely ambiguous or overlap heavily with other accounts (e.g., distinguishing "Branding" expenses from general "Marketing Promo Costs"). 

---

## 5. Discussion

### 5.1 Strengths of the Approach
1.  **Robust to Noise:** The implementation of the `ColumnTransformer` combined with tailored n-gram ranges for different text types allowed the model to extract highly predictive signals from messy, unstructured text.
2.  **Algorithmic Imbalance Handling:** Relying on `class_weight='balanced'` inside an SVM successfully mitigated the massive dataset skew without the need for complex, computationally expensive resampling techniques like SMOTE.

### 5.2 Limitations
1.  **Contextual Understanding:** TF-IDF is a frequency-based statistical method; it does not understand the semantic meaning of words. If a new transaction uses a synonym the model hasn't seen before, it may misclassify it.
2.  **Scalability of SVM:** While the linear kernel SVM trained efficiently, if the dataset grows to millions of rows, SVMs become computationally expensive to train compared to gradient boosting frameworks.

### 5.3 Ideas for Improvement
With more time and computational resources, the following improvements could be made:
1.  **Semantic Embeddings:** Replacing TF-IDF with pre-trained word embeddings (like Word2Vec) or modern Large Language Model (LLM) embeddings (like BERT). This would allow the model to understand the *meaning* of item descriptions rather than just counting word occurrences.
2.  **Ensemble Methods:** Blending the predictions of the SVM with a tree-based model (like XGBoost) could capture both linear boundaries in the text and non-linear interactions in the numeric amounts.

### 5.4 Business Considerations for Deployment
If deployed into a production financial system, this model should be implemented as a "Human-in-the-Loop" assistant rather than a fully autonomous agent. 
*   **Confidence Thresholds:** The system should automatically categorize transactions where the SVM's prediction confidence is high (e.g., >90%). 
*   **Manual Review:** For transactions where the model is uncertain (low confidence score), the transaction should be flagged and routed to a human accountant for manual review. Over time, these manual corrections can be fed back into the model to continuously improve its accuracy on ambiguous edge cases.

### 5.5 Alternative Approaches Evaluated
During the development phase, three other prominent algorithms were tested before finalizing the Support Vector Machine:
1.  **Logistic Regression:** Tested as an initial baseline. While computationally lightweight, it struggled to capture the complex boundaries required to differentiate between closely related textual descriptions, resulting in subpar overall accuracy.
2.  **XGBoost:** A highly optimized tree-based pipeline. While XGBoost is generally exceptionally powerful, its `multi:softprob` objective algorithm scales poorly with the extreme cardinality of the `accountName` target. Furthermore, tree-based models computationally struggle with massive, sparse TF-IDF feature spaces. Despite heavy optimizations (using `tree_method='hist'`), it achieved 85.24% accuracy but required significantly longer training times compared to the linear SVM.
3.  **CatBoost:** Another advanced gradient boosting framework . While CatBoost handles raw categorical data exceptionally well, passing it a massive, pre-computed sparse TF-IDF matrix completely negated its internal optimizations. Its underlying symmetric decision trees struggled to find meaningful splits across thousands of near-zero sparse features, leading to severe underperformance (only 65.50% accuracy) compared to the linear SVM.
