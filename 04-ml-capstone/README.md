# 04-ml-capstone

Capstone project refactoring four course appendix scripts (binary logistic regression, multiclass OvO logistic regression, decision tree with CV tuning, and SVM with CV tuning) into a reproducible, configurable Python mini-project.

> **Note:** Dataset `.mat` files are **not** included. You must supply your own `.mat` file via `--mat-path`.

---

## Directory layout

```
04-ml-capstone/
├── README.md
├── requirements.txt
├── reports/          # auto-created at run time; stores metrics.json, summary.csv, plots
├── src/
│   ├── __init__.py
│   ├── inspect_mat.py          # CLI: inspect .mat file keys and shapes
│   ├── data.py                 # configurable .mat loader
│   ├── metrics.py              # accuracy / precision / recall / F1 / confusion matrix
│   ├── plots.py                # save confusion-matrix heatmap and decision-tree plot
│   ├── logreg_newton_binary.py # binary logistic regression (Newton / Hessian update)
│   ├── logreg_ovo_multiclass.py# multiclass OvO logistic regression (majority vote)
│   ├── decision_tree_tuning.py # decision tree + 5-fold CV grid search
│   ├── svm_tuning.py           # SVM + StratifiedKFold CV + held-out test evaluation
│   └── run.py                  # unified CLI entrypoint
└── tests/
    └── test_smoke.py           # smoke tests (no .mat data required)
```

---

## Installation

```bash
# from the repo root
pip install -r 04-ml-capstone/requirements.txt
```

---

## Inspecting a `.mat` file

Before running experiments you can discover which variable keys are stored in a `.mat` file and what shapes they have:

```bash
python 04-ml-capstone/src/inspect_mat.py --mat-path /path/to/dataset.mat
```

Sample output:
```
Keys in /path/to/dataset.mat
  '__header__'        : (metadata, skipped)
  '__version__'       : (metadata, skipped)
  '__globals__'       : (metadata, skipped)
  'data'              : shape=(150, 5)  dtype=float64

Suggested defaults:
  --mat-key  data
  Array has 5 columns. Label is likely the first or last column.
  --label-col last   (column index 4)
  --label-col first  (column index 0)
```

---

## Running experiments

### Unified CLI (`run.py`)

```
python 04-ml-capstone/src/run.py \
    --mat-path /path/to/dataset.mat \
    [--mat-key KEY]          # defaults to first non-metadata key
    [--label-col {first,last,INT}]  # default: last
    [--model {logreg_binary,logreg_ovo,decision_tree,svm}]
    [--tune]                 # enable CV hyper-parameter search
    [--test-size 0.2]        # train/test split ratio
    [--random-state 42]
    [--run-id RUN_ID]        # sub-folder name under reports/; auto-generated if omitted
    [--repeats 10]           # (logreg_binary) number of random splits
    [--max-iter 300]         # (logreg_binary / logreg_ovo)
```

### Examples

```bash
# inspect first
python 04-ml-capstone/src/inspect_mat.py --mat-path dataset.mat

# binary logistic regression with Newton updates, 10 random splits
python 04-ml-capstone/src/run.py --mat-path dataset.mat --mat-key X \
    --label-col last --model logreg_binary --repeats 10

# multiclass OvO logistic regression
python 04-ml-capstone/src/run.py --mat-path dataset.mat \
    --model logreg_ovo --tune

# decision tree with 5-fold CV grid search
python 04-ml-capstone/src/run.py --mat-path dataset.mat \
    --model decision_tree --tune

# SVM with StratifiedKFold CV (fixes evaluation leakage)
python 04-ml-capstone/src/run.py --mat-path dataset.mat \
    --model svm --tune --test-size 0.2
```

---

## Outputs

Every run writes its results to `04-ml-capstone/reports/<run-id>/`:

| File | Contents |
|------|----------|
| `metrics.json` | accuracy, precision, recall, F1 (macro/micro/weighted), best params |
| `summary.csv` | one row per run appended for easy comparison |
| `confusion_matrix.png` | heatmap of the test-set confusion matrix |
| `tree.png` | decision-tree structure (decision_tree model only) |

---

## Reproducibility

Pass `--random-state INT` to fix all random seeds. Default is `42`.
