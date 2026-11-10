# Beyond BMI: A Leakage-Aware Evaluation of Lifestyle-Based Machine Learning for Obesity Classification

**ANLY 530 | Applied Machine Learning**  
**Harrisburg University of Science and Technology**  
**KA1: Research Design and Feasibility**

## Research Overview

High predictive accuracy does not necessarily indicate meaningful learning. This concern is particularly relevant to obesity classification, where models trained with height and weight may simply reconstruct BMI-related categories rather than discover independently informative behavioral patterns.

This study investigates whether dietary habits, physical activity, and demographic characteristics can classify seven obesity-level categories when height, weight, and derived BMI are excluded.

The project emphasizes scientific validity, transparent evaluation, and reproducibility rather than predictive accuracy alone.

## Research Questions

**Primary research question:**  
How effectively can lifestyle, dietary, and demographic characteristics classify seven obesity-level categories without anthropometric predictors?

**Secondary research questions:**

1. How does classification performance change when height and weight are removed?
2. How sensitive are model results to validation methods and potentially dependent synthetic observations?
3. Which obesity categories are most frequently misclassified?

## Research Contribution

The study proposes a leakage-aware evaluation framework for lifestyle-based obesity classification.

Its contribution is methodological: examining whether machine-learning performance reflects meaningful information beyond BMI-related label reconstruction.

The anticipated audience includes public-health researchers, healthcare administrators, and applied machine-learning investigators.

## Dataset and Provenance

**Dataset:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition

**Source:** UCI Machine Learning Repository

**Dataset URL:**  
https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

**Dataset DOI:** https://doi.org/10.24432/C5H31Z

**Observations:** 2,111  
**Predictor variables:** 16  
**Target variable:** NObeyesdad  
**Classification task:** Seven-class obesity-level classification

The dataset contains information associated with individuals from Mexico, Peru, and Colombia. Approximately 23% of observations were directly collected, while 77% were synthetically generated.

The absence of verified record-level provenance restricts the ability to establish independent real-world generalization.

## Methodological Framework

Three feature representations will be examined:

| Feature Set | Description | Research Purpose |
|---|---|---|
| A | Height and weight | Anthropometric reference and label-reconstruction assessment |
| B | Lifestyle and demographic features, excluding height and weight | Primary classification experiment |
| C | Behavioral features excluding height, weight, age, gender, and family history | Behavior-only sensitivity analysis |

### Machine-Learning Methods

The proposed models are:

- Majority-class DummyClassifier (baseline)
- Logistic regression
- k-nearest neighbors (kNN)
- Random forest

These approaches provide interpretable, distance-based, and nonlinear comparisons.

### Preprocessing

Preprocessing will include:

- Data-type and missing-value inspection
- Duplicate and near-duplicate assessment
- Numerical imputation when needed
- Categorical encoding
- Feature scaling where appropriate
- Training-fold-only preprocessing through scikit-learn pipelines

### Validation Strategy

A stratified 80/20 training-test partition is proposed.

Model comparison and hyperparameter selection will use stratified five-fold cross-validation within the training partition.

The held-out test set will remain untouched until model selection is complete.

### Evaluation Metrics

**Primary metric:** Macro-F1 score

**Secondary metrics:**

- Balanced accuracy
- Per-class precision and recall
- Confusion matrices
- Ordered-category error analysis
- Validation variability

Sensitivity analyses will examine potential dependence among synthetic observations and the consequences of different feature representations.

## Reproducibility and Software

The project will use:

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook
- ucimlrepo

The repository will contain a requirements file and executable exploratory notebook.

### Planned Repository Structure

```text
obesity-ml-research/
├── README.md
├── .gitignore
├── requirements.txt
└── 01_exploratory_data_analysis.ipynb
```

### Data Retrieval

The dataset can be obtained directly from UCI using Python:

```python
from ucimlrepo import fetch_ucirepo

obesity = fetch_ucirepo(id=544)

X = obesity.data.features
y = obesity.data.targets

print(X.shape)
print(y.value_counts())
```

After the requirements file has been added, dependencies can be installed using:

```bash
pip install -r requirements.txt
```

## Research Timeline

| Milestone | Planned Activities |
|---|---|
| Week 10 | Research protocol, dataset exploration, and repository setup |
| Week 12 | Model implementation, comparison, and validation |
| Week 13 | Final conference paper, reproducibility review, and five-minute presentation |

## Anticipated Outputs

The study is expected to produce:

- An exploratory data-quality report
- A comparative model-performance table
- Cross-validation summaries
- A held-out evaluation
- Confusion matrices
- A feature-representation sensitivity analysis
- A discussion of methodological limitations

These outputs are planned; empirical findings have not yet been established.

## Limitations and Ethical Considerations

Important limitations include the dataset's substantial synthetic component, potentially dependent records, BMI-related target construction, self-reported information, and limited geographic representativeness.

Findings will not be interpreted as evidence of clinical readiness or used for individual diagnosis or treatment decisions.

The repository will not contain private or restricted data.

## Research Contribution Statement

This is an individual research project. The student researcher is responsible for the research design, literature review, data exploration, methodological decisions, code execution, evaluation, interpretation, and documentation.

## Reference

Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief, 25*, 104344. https://doi.org/10.1016/j.dib.2019.104344

---

**Project status:** Milestone 1 — Research Design and Feasibility.
