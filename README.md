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
- Evaluate the trade-off between predictive performance and the amount of training data retained.

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

The repository files are organized at the root level:

- `Waveforms_project.ipynb` — notebook containing the code and experiments.
- `Waveforms_Project.pdf` — project report.
- `waveform.data` — dataset file expected by the notebook.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes common temporary and environment files.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sultan-Haider-Cheema/KNN_waveform.git
cd KNN_waveform
```

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

The notebook loads the dataset using:

```python
pd.read_csv('waveform.data', header=None, sep=',')
```

Keep `waveform.data` in the repository root alongside `Waveforms_project.ipynb`. This allows the notebook to load the dataset using its existing relative path, provided you launch Jupyter from the repository root.

If you store the dataset elsewhere, update the path in the notebook accordingly.

### 5. Run the notebook

From the repository root, run:

```bash
jupyter notebook
```

Open `Waveforms_project.ipynb` and execute the cells in order. Use **Restart Kernel and Run All Cells** to verify the notebook from a clean state.

## Results and Evaluation

The experiments evaluate the performance of k-Nearest Neighbors (kNN) and compare it with Decision Tree, Random Forest, and RBF-kernel Support Vector Machine (SVM) classifiers.

The analysis focuses on:

- **Hyperparameter tuning:** Examining how the choice of `k` affects classification performance.
- **Generalization behaviour:** Comparing training and test errors across different training-set sizes.
- **Classifier comparison:** Evaluating the predictive performance of different machine learning algorithms.
- **Data reduction:** Investigating how Condensed Nearest Neighbour (CNN) and Reduced Nearest Neighbour (RNN)-style methods affect the number of retained training instances and classification performance.

The experimental results, visualizations, and detailed comparisons are available in the Jupyter Notebook and the accompanying project report.

## Notes and Limitations

- Results may vary depending on the dataset version, train-test split, and experimental configuration.
- Feature standardization and hyperparameter selection should be performed using training data to avoid data leakage.
- Data-reduction techniques can reduce the size of the training set, but their impact on predictive performance should also be considered.
- The dataset is included for experimental use; users should check its original licensing and redistribution terms before reusing or redistributing it.

## Project Report

See `Waveforms_Project.pdf` for the accompanying report.

## Author

**Muhammad Sultan Haidar Cheema**

Master's student in Machine Learning and Data Mining  
Université Jean Monnet, Saint-Étienne, France

## Repository

[View the project on GitHub](https://github.com/Sultan-Haider-Cheema/KNN_waveform)
