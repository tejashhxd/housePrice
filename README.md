# House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)

A reproducible machine-learning notebook for predicting residential sale prices using the Kaggle *House Prices: Advanced Regression Techniques* dataset.

The main notebook cleans raw tabular data through scikit-learn pipelines, compares a Random Forest and a linear Support Vector Regressor with five-fold cross-validation, selects the lower-RMSLE model, retrains it on every labeled row, and exports a Kaggle-compatible submission file.

## Project Overview

This repository contains a two-stage modeling workflow:

1. **Model evaluation**
   - Load `train.csv` and `test.csv`
   - Validate the training and test feature columns
   - Separate `SalePrice` from the predictors and exclude `Id`
   - Impute and encode features inside reusable scikit-learn pipelines
   - Compare Random Forest and SVR models using RMSLE
   - Run an additional fixed 20% holdout sanity check

2. **Submission generation**
   - Select the model with the lowest mean cross-validated RMSLE
   - Refit it on all 1,460 training observations
   - Predict `SalePrice` for all 1,459 test observations
   - Write the result to `submission.csv`

## Repository Contents

| File | Description |
| --- | --- |
| `HousePrice_Submission.ipynb` | Main reproducible preprocessing, evaluation, training, and submission notebook |
| `train.csv` | 1,460 labeled rows with 79 predictors and `SalePrice` |
| `test.csv` | 1,459 unlabeled rows with 79 predictors |
| `submission.csv` | Generated predictions in Kaggle submission format |
| `sample_submission.csv` | Reference format containing `Id` and `SalePrice` columns |
| `data_description.txt` | Definitions and allowed values for the dataset fields |
| `Untitled.ipynb` | Additional exploratory notebook |
| `Untitled1.ipynb` | Additional exploratory notebook |

## Data Description

The data describes homes in Ames, Iowa. It includes:

- **Property attributes:** lot size, dwelling type, house style, and street access
- **Construction details:** year built, roof, foundation, masonry, and external materials
- **Interior features:** room counts, floor area, condition, quality, heating, and fireplaces
- **Garage and exterior features:** garage capacity and finish, porch areas, fences, and pools
- **Sale details:** sale type, condition, month, and year

The target variable is `SalePrice`. It is modeled on a log scale with `log1p(SalePrice)` and converted back to dollars with `expm1` for predictions. Using RMSLE emphasizes relative rather than absolute errors and matches the Kaggle competition objective.

Each missing categorical value is represented by a `Missing` category. Numeric missing values are median-imputed, and an indicator feature records which numeric values were missing. Categorical features are one-hot encoded, while numeric features are standardized.

## Results

The notebook's recorded five-fold cross-validation results are:

| Model | Mean CV RMSLE | Standard Deviation |
| --- | ---: | ---: |
| Random Forest | 0.142684 | 0.009235 |
| Linear SVR | 0.159167 | 0.032084 |

The Random Forest had the lower mean cross-validated RMSLE and was selected for the final submission.

The fixed holdout produced an additional reference estimate:

| Model | Holdout RMSLE |
| --- | ---: |
| Linear SVR | 0.137411 |
| Random Forest | 0.146442 |

The holdout is for comparison only. The model for the final submission is chosen by mean cross-validation RMSLE rather than holdout performance.

## Getting Started

### 1. Create a virtual environment

Run the following commands from this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, you can activate with Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

```powershell
python -m pip install jupyter numpy pandas scikit-learn
```

The workflow uses NumPy, pandas, Jupyter, and scikit-learn. A machine-learning environment such as Conda or JupyterLab can also be used.

### 3. Start Jupyter

```powershell
jupyter notebook
```

Open `HousePrice_Submission.ipynb` in Jupyter and run every cell with **Kernel > Restart & Run All**.

## Notebook Workflow

The notebook executes these stages:

1. Set `RANDOM_STATE = 42` and `DATA_DIR = Path('.')`
2. Read `train.csv` and `test.csv`
3. Remove `SalePrice` and `Id` from the model feature set
4. Check for mismatched training and test feature columns
5. Construct separate numeric and categorical preprocessing pipelines
6. Wrap each estimator in `TransformedTargetRegressor`
7. Compare five-fold RMSLE scores
8. Display a fixed 20% holdout comparison
9. Refit the cross-validation winner on all labeled rows
10. Predict 1,459 test rows and validate the submission shape
11. Save `submission.csv`

For a data directory other than the notebook directory, update:

```python
DATA_DIR = Path(".")
```

For example:

```python
DATA_DIR = Path(r"D:\data\house-price")
```

## Modeling Pipeline

The reusable pipeline has two feature branches:

```text
Raw features
├── Numeric columns
│   ├── Median imputation
│   ├── Missing-value indicators
│   └── Standard scaling
└── Categorical columns
    ├── Constant imputation ("Missing")
    └── One-hot encoding (unknown categories ignored)
```

The complete model also applies target transformation:

```text
SalePrice -> log1p(SalePrice) -> model.fit()
model.predict() -> expm1(prediction) -> SalePrice
```

Preprocessing is learned inside cross-validation folds and is therefore not fitted on validation or test rows before those rows are processed.

## Random Forest Configuration

```python
RandomForestRegressor(
    n_estimators=500,
    max_features=1.0,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
```

## Linear SVR Configuration

```python
SVR(
    kernel="linear",
    C=100,
    epsilon=0.1,
)
```

Both models are starting configurations rather than hyperparameter-optimized solutions.

## Submission Format

`submission.csv` contains exactly the two columns required by Kaggle:

```csv
Id,SalePrice
1461,124852.44239366023
1462,154887.11377337942
1463,177005.46159842564
```

The file must contain 1,459 rows, in the same ID order as `test.csv`, with no missing predictions.

Upload `submission.csv` to the Kaggle competition page under **Submit Predictions**.

## Reproducibility

- `RANDOM_STATE = 42` controls the holdout split and Random Forest randomness
- Five-fold cross-validation uses the same competition metric for both estimators
- Scikit-learn preprocessors are recreated for each model
- The output submission is generated locally from `test.csv`
- The notebook validates feature-column names and output shape before writing the file

Results can vary slightly when using notably different scikit-learn or pandas versions. Pinning exact package versions is recommended for a fully reproducible environment.

## Possible Improvements

- Tune Random Forest hyperparameters with `RandomizedSearchCV`
- Evaluate gradient boosting, HistGradientBoosting, XGBoost, or CatBoost
- Add ordinal encoding for quality and condition fields
- Engineer living-area, total-area, property-age, and renovation features
- Use target or feature transformations for skewed numerical variables
- Compare alternatives with nested cross-validation
- Ensemble several independently validated models
- Track experiments and package versions with requirements or environment files

## License

No repository license is currently included. Add a license before distributing or publishing the project if a specific reuse policy is intended.
