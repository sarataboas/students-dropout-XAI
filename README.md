
# Explainable AI Techniques on Students Dropout and Academic Performance
Assignment for Advanced Topics in Machine Learning, 1st Year, 1st Semester, Master degree in Artificial Intelligence 

## Project Overview

This project investigates the application of Explainable Artificial Intelligence (XAI) techniques to the problem of student dropout prediction in higher education. Using the Students’ Dropout and Academic Performance dataset, we analyze how different explainability methods can be applied throughout the machine learning pipeline to improve model transparency, trust, and actionability.

The study combines:

- Pre-modelling explainability (Exploratory Data Analysis and dataset summarization),

- In-modelling explainability using interpretable (glass-box) models,

- Post-modelling explainability applied to a high-performing black-box model.

The goal is not only to achieve good predictive performance, but also to understand why students are predicted to drop out and how such predictions can support targeted institutional interventions.



**Authors**

- Guilherme Oliveira

- Ricardo Costa

- Sara Táboas

*Faculty of Sciences, University of Porto*


## Dataset

The project uses the Students’ Dropout and Academic Performance dataset, publicly available from the UCI Machine Learning Repository.

- Number of students: 4,424

- Features: 36

- Target variable: Academic status (Dropout, Enrolled, Graduate)

For modeling purposes, the task is framed as a binary classification problem (Dropout vs. Graduate), with Enrolled students used later for practical intervention analysis


## Methodology
- Pre-Modelling Explainability

    - Exploratory Data Analysis (EDA)

    - Data Summarization using PCA and t-SNE (exploratory only)


- In-Modelling Explainability: These models provide transparent global explanations through their internal structure.

    - Decision Tree classifier

    - Explainable Boosting Machine (EBM)

- Post-Modelling Explainability: A black-box XGBoost classifier is used as the reference predictive model. The following XAI techniques are applied:

    - Simplification-based: Surrogate Decision Tree

    - Feature-based: Global feature importance, ALE, SHAP

    - Example-based: Anchors, Counterfactual explanations


## Practical Application

- Dropout risk prediction for Enrolled students

- SHAP-based diagnosis

- Counterfactual simulations to assess actionable interventions

## Requirements
The project was developed using Python 3.10+. You can install the required packages with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run
1. Clone the repository:
```bash
git clone <repository-url>
```

2. Open the notebook
```bash
jupyter notebook student_dropout_xai.ipynb
```



## Results Summary

- Academic performance, particularly second-semester outcomes, is the strongest predictor of dropout.

- Financial variables (e.g., tuition fee status) play a critical role for academically capable students.

- Different XAI methods provide complementary insights:

- Glass-box models offer simplicity and stability,

- Feature-based methods capture non-linear effects,

- Example-based methods enable actionable interventions.