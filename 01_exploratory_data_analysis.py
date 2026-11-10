
# ============================================================
# ANLY 530 | KA1 Milestone 1
# Research Design and Feasibility
#
# Beyond BMI: A Leakage-Aware Evaluation of Lifestyle-Based
# Machine Learning for Obesity Classification
#
# Purpose:
# Conduct preliminary exploratory data analysis and assess
# potential sources of target leakage and data dependence.
#
# Dataset: UCI Obesity Levels Dataset (ID: 544)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo


# ------------------------------------------------------------
# 1. DATA ACQUISITION AND PROVENANCE
# ------------------------------------------------------------

print("Loading the UCI Obesity Levels dataset...")

obesity = fetch_ucirepo(id=544)

X = obesity.data.features.copy()
y = obesity.data.targets.copy()

data = pd.concat([X, y], axis=1)

print("\nDATASET OVERVIEW")
print("Number of observations:", data.shape[0])
print("Number of predictor variables:", X.shape[1])
print("Total columns including target:", data.shape[1])

print("\nVariable names:")
print(data.columns.tolist())


# ------------------------------------------------------------
# 2. INITIAL DATA INSPECTION
# ------------------------------------------------------------

print("\nFIRST FIVE OBSERVATIONS")
print(data.head())

print("\nDATA TYPES")
print(data.dtypes)

print("\nNUMERICAL SUMMARY")
print(data.describe().round(2))


# ------------------------------------------------------------
# 3. MISSING DATA ASSESSMENT
# ------------------------------------------------------------

print("\nMISSING DATA ASSESSMENT")

missing_counts = data.isna().sum()
missing_percentages = data.isna().mean() * 100

missing_summary = pd.DataFrame({
    "Missing_Count": missing_counts,
    "Missing_Percentage": missing_percentages.round(2)
})

print(missing_summary)


# ------------------------------------------------------------
# 4. DUPLICATE RECORD ASSESSMENT
# ------------------------------------------------------------

print("\nDUPLICATE RECORD ASSESSMENT")

exact_duplicates = data.duplicated().sum()

print("Number of exact duplicate records:", exact_duplicates)

print(
    "Duplicate screening is preliminary. "
    "It does not establish independence among synthetic records."
)


# ------------------------------------------------------------
# 5. TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------------------

target_column = "NObeyesdad"

print("\nTARGET CLASS DISTRIBUTION")

class_counts = data[target_column].value_counts()
class_percentages = (
    data[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

class_summary = pd.DataFrame({
    "Frequency": class_counts,
    "Percentage": class_percentages
})

print(class_summary)

plt.figure(figsize=(11, 5))

sns.countplot(
    data=data,
    x=target_column,
    order=data[target_column].value_counts().index
)

plt.title("Distribution of Obesity Classification Categories")
plt.xlabel("Obesity Category")
plt.ylabel("Number of Observations")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. FEATURE REPRESENTATION AND TARGET LEAKAGE
# ------------------------------------------------------------

print("\nFEATURE REPRESENTATION ASSESSMENT")

# Anthropometric predictors may reconstruct BMI-based labels.
anthropometric_features = ["Height", "Weight"]

# Primary experiment: exclude height and weight.
lifestyle_features = X.drop(
    columns=anthropometric_features,
    errors="ignore"
)

# Sensitivity experiment: behavior-oriented predictors only.
behavior_only_features = lifestyle_features.drop(
    columns=[
        "Age",
        "Gender",
        "family_history_with_overweight"
    ],
    errors="ignore"
)

print("Original number of predictors:", X.shape[1])

print(
    "Predictors excluding height and weight:",
    lifestyle_features.shape[1]
)

print(
    "Behavior-only predictor count:",
    behavior_only_features.shape[1]
)

print("\nPrimary lifestyle and demographic predictors:")
print(lifestyle_features.columns.tolist())

print("\nBehavior-only predictors:")
print(behavior_only_features.columns.tolist())


# ------------------------------------------------------------
# 7. REPEATED FEATURE PATTERN ASSESSMENT
# ------------------------------------------------------------

print("\nREPEATED FEATURE PATTERN ASSESSMENT")

repeated_patterns = lifestyle_features.duplicated().sum()

print(
    "Repeated lifestyle/demographic feature combinations:",
    repeated_patterns
)

print(
    "Identical feature combinations do not necessarily "
    "represent duplicate individuals or synthetic dependence."
)


# ------------------------------------------------------------
# 8. NUMERICAL VARIABLE DISTRIBUTIONS
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=np.number
).columns

print("\nNUMERICAL PREDICTORS")
print(numeric_features.tolist())

if len(numeric_features) > 0:
    X[numeric_features].hist(
        figsize=(12, 9),
        bins=20
    )

    plt.suptitle(
        "Exploratory Distributions of Numerical Predictors",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. PRELIMINARY DATA QUALITY SUMMARY
# ------------------------------------------------------------

print("\nPRELIMINARY DATA QUALITY SUMMARY")

print("Observations:", len(data))
print("Missing cells:", int(data.isna().sum().sum()))
print("Exact duplicate records:", int(exact_duplicates))
print("Target classes:", data[target_column].nunique())

print(
    "Approximately 77% of the source dataset is reported "
    "as synthetic in its documentation."
)

print(
    "Row-level original/synthetic provenance cannot be "
    "verified from the distributed dataset."
)


# ------------------------------------------------------------
# 10. RESEARCH DESIGN AND NEXT STEPS
# ------------------------------------------------------------

print("\nRESEARCH DESIGN: PLANNED METHODS")

print("Baseline: Majority-class DummyClassifier")
print("Candidate model 1: Logistic Regression")
print("Candidate model 2: k-Nearest Neighbors")
print("Candidate model 3: Random Forest")

print("\nPLANNED VALIDATION")

print("Stratified 80/20 train-test partition")
print("Five-fold stratified cross-validation")
print("Training-fold-only preprocessing")
print("Primary evaluation metric: Macro-F1")
print("Secondary evaluation: Balanced accuracy")
print("Additional evaluation: Per-class metrics")
print("Sensitivity analysis: Potential synthetic dependence")

print("\nMILESTONE 1 STATUS")
print("Exploratory analysis script prepared.")
print("Predictive models have not been trained.")
print("Model performance results have not been established.")
print("No clinical validity is claimed.")
