# smartphone-addiction-analysis-ml
Smartphone Addiction Level Analysis & Classification

This repository contains an exploratory data analysis (EDA) and machine learning project aimed at analyzing, scoring, and classifying smartphone addiction levels based on user habits and behavioral indicators.

Note on Language & Presentation:

The internal code comments/headers and the accompanying presentation slides (project_presentation.pptx / project_presentation.pdf) are prepared in Turkish as this project was originally delivered for an academic coursework/seminar. A complete English summary of the methodology, pipeline, and findings is documented below.

📌 Project Overview

Excessive smartphone usage and digital dependency have become widespread concerns affecting productivity, mental well-being, and daily routines. This project analyzes behavioral patterns to categorize smartphone addiction severity into stages/scores using supervised machine learning algorithms.

Key Objectives:

Perform data preprocessing, feature encoding, and missing value imputation.

Analyze correlations between usage metrics (e.g., screen time, notification frequency) and addiction degrees.

Train and evaluate supervised classification models (KNN, SVM).

Benchmark model performance using standard evaluation metrics.

📊 Dataset

Source: Kaggle (Smartphone Addiction / User Behavior Dataset)

File Name: smartphone_addiction_dataset.csv

Key Indicators:

Daily screen time and active device hours

Notification check frequency / device pickups

Primary usage categories (social media, gaming, communication, productivity)

Target addiction scale / categorical classification score

⚙️ Tech Stack & Libraries

Language: Python

Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

🔬 Methodology & Workflow

Exploratory Data Analysis (EDA): Feature distribution checks, outlier detection, and correlation matrices.

Feature Engineering & Preprocessing: Data standardization/normalization and categorical variable encoding.

Model Training & Evaluation:

Evaluated algorithms such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).

Assessed accuracy, precision, recall, and confusion matrices across train/test splits.

📈 Results & Findings

The best-performing model achieved high classification performance across user addiction tiers.

Usage duration during late hours and high notification check frequency showed the strongest positive correlation with elevated addiction scores.

📂 Project Structure

├── smartphone_addiction_dataset.csv        # Kaggle dataset
├── project_presentation.pptx               # Presentation slides (Turkish)
├── project_presentation.pdf                # PDF version for quick preview
├── main.py                                 # Python code
└── README.md                               # Project documentation


🚀 How to Run

Clone the repository:

git clone https://github.com/yunusemretemel/smartphone-addiction-analysis-ml.git
cd smartphone-addiction-analysis-ml


Install dependencies:

pip install -r requirements.txt


Launch the notebook:

jupyter notebook smartphone_addiction_analysis.ipynb
