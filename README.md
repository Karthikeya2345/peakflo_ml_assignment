# Financial Transaction Classification Pipeline

This repository contains the end-to-end Machine Learning pipeline (`peakflo_ml_assignment.ipynb`) designed to classify financial transaction records into their correct accounts. The model utilizes TF-IDF and a Support Vector Machine (SVM) to handle high-cardinality textual data and severe class imbalances, achieving 87.9% overall accuracy.

## Files
- `peakflo_ml_assignment.ipynb`: The primary notebook containing the full pipeline (Data loading, EDA, feature engineering, model training, and evaluation).
- `Execution_Report.md`: The comprehensive written assignment report.
- `requirements.txt`: The Python dependencies needed to run the notebook.

## How to Run

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `peakflo_ml_assignment.ipynb` in your preferred environment (Jupyter Notebook, JupyterLab, or VSCode).
3. Run all cells sequentially. The notebook will automatically download the dataset (or generate a fallback dummy dataset), clean the data, train the SVM, and output the performance metrics and confusion matrix.
