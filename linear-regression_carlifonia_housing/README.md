# California Housing — Linear Regression from Scratch

A end-to-end machine learning project implementing linear regression using only NumPy on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset). Built to understand the fundamentals of the ML pipeline without relying on high-level abstractions.

---

## Project Overview

This project walks through the full supervised learning pipeline:

- **Exploratory Data Analysis (EDA)** — distributions, correlations, and outlier inspection
- **Feature Engineering** — transformations, interaction terms, and scaling
- **Model Implementation** — gradient descent and the normal equation built from scratch with NumPy
- **Validation** — results benchmarked against scikit-learn's `LinearRegression`
- **Diagnostics** — residual analysis, multicollinearity checks (VIF), and model limitations

Key issues encountered and resolved during the project:
- Data leakage (scaling before train/test split)
- Multicollinearity among features
- The dummy variable trap
- Rank deficiency in the design matrix

---

## Repository Structure

```
california-housing-regression/
├── data/                  # Dataset files (not tracked by Git — see note below)
├── notebooks/             # Jupyter notebooks
│   └── linear_regression.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note on data:** The raw dataset is not committed to this repo. It is loaded directly via `sklearn.datasets.fetch_california_housing()` inside the notebook, so no manual download is needed.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/california-housing-regression.git
cd california-housing-regression
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook notebooks/
```

---

## Dependencies

See `requirements.txt`. Core libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Linear algebra, gradient descent |
| `pandas` | Data manipulation and EDA |
| `matplotlib` | Visualizations |
| `seaborn` | Statistical plots |
| `scikit-learn` | Dataset loading and validation baseline |
| `statsmodels` | VIF / multicollinearity diagnostics |
| `jupyter` | Notebook environment |

---

## Results

The from-scratch implementation achieves results consistent with scikit-learn's `LinearRegression`, confirming correctness of the gradient descent and normal equation implementations.

| Metric | From Scratch | scikit-learn |
|---|---|---|
| R² (test) | — | — |
| RMSE (test) | — | — |

> Fill in your actual numbers above before publishing.

---

## Limitations

- Linear regression assumes a linear relationship between features and the target; the California Housing data has notable non-linearities.
- Outliers in `AveRooms` and `AveOccup` affect coefficient estimates.
- Geographic features (`Latitude`, `Longitude`) are used as raw inputs rather than being modelled spatially.
- No regularisation (Ridge/Lasso) is applied; the model may overfit on noisy features.

---

## Author

**Frank**  
[GitHub](https://github.com/YOUR_USERNAME)
