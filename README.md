# Waveform Classification with kNN and Data Reduction

A machine learning project exploring k-Nearest Neighbors (kNN), hyperparameter tuning, generalization behaviour, instance reduction, and comparison with other classifiers on the Waveform dataset.

## Project Overview

This project studies how the number of neighbors, training-set size, and data-reduction methods affect classification performance on a three-class Waveform classification task.

## Objectives

- Tune the number of neighbors (`k`) using cross-validation.
- Explore training and test error across different training-set sizes.
- Implement Condensed Nearest Neighbour (CNN) instance selection.
- Implement a Reduced Nearest Neighbour (RNN)-style reduction experiment.
- Compare kNN with Decision Tree, Random Forest, and RBF-kernel Support Vector Machine (SVM).
- Consider the trade-off between predictive performance and the amount of training data retained.

## Methods

### Preprocessing
- Load the Waveform dataset from `waveform.data`.
- Split the data into training and test sets using a fixed random seed.
- Standardize features using `StandardScaler`.

### Experiments
- kNN hyperparameter search with 5-fold cross-validation.
- Error analysis across different training-set sizes.
- CNN and RNN-style data-reduction experiments.
- Comparison of kNN, Decision Tree, Random Forest, and RBF SVM.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## Repository Contents

- `Waveforms_project.ipynb` — notebook containing the code and experiments.
- `Waveforms_Project.pdf` — project report.
- `waveform.data` — dataset file expected by the notebook.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes common temporary and environment files.

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Check the dataset location

The notebook currently loads the dataset using:

```python
pd.read_csv("waveform.data", header=None, sep=",")
```

Therefore, place `waveform.data` in the same directory from which you launch the notebook (normally the repository root), or update the path in the notebook to match your file location.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `Waveforms_project.ipynb` and run the cells in order. In Jupyter, use **Restart Kernel and Run All Cells** to verify the notebook from a clean state.

## Results

The notebook prints and plots the experimental results, including cross-validation scores, classification accuracy, data-reduction statistics, and classifier comparisons. Refer to the notebook and accompanying PDF report for the measured values.

No performance numbers are stated here so that the README does not report unverified results. If you add a results table, copy the values from a successful, reproducible run and clearly identify which split and metric each value represents.

## Notes and Limitations

- Use the same dataset version as the original experiments.
- Keep preprocessing and model selection within the training data; use the held-out test set for final evaluation.
- Data-reduction methods should be assessed using both the number of retained samples and predictive performance.
- Check the dataset's terms before redistributing it publicly.

## Project Report

See `Waveforms_Project.pdf` for the accompanying report.
