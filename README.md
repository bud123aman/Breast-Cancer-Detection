# Breast Cancer Detection Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bud123aman/Breast-Cancer-Detection/blob/main/Breast_Cancer_Detection.ipynb)

This project focuses on developing a machine learning model to **detect breast cancer** using a Logistic Regression algorithm. Early detection is crucial for effective treatment, and this model aims to provide a reliable tool for classification.

---

## Project Overview

This repository contains the code for a breast cancer detection system built with Python and popular machine learning libraries. The model is trained on a dataset containing various features related to breast cancer and is designed to classify a tumor as either benign or malignant.

---

## Key Features

* **Data Preprocessing:** Handles data loading and prepares it for model training.
* **Logistic Regression Model:** Implements a Logistic Regression classifier for robust predictions.
* **Model Evaluation:** Utilizes a **Confusion Matrix** to visualize performance and calculates **accuracy** using both a standard test set and **K-Fold Cross-Validation** for more reliable performance estimation.

---

## Technologies Used

* Python
* `pandas` for data manipulation
* `scikit-learn` for machine learning algorithms and model evaluation

---

## Model Performance

The model demonstrates strong performance in detecting breast cancer, with a particular focus on **accuracy** as a key evaluation metric. Accuracy measures the proportion of total predictions that were correct.

* **Overall Accuracy:** The model achieved an impressive accuracy, indicating its ability to correctly classify both benign and malignant cases.
    * **Accuracy (K-Fold Cross-Validation):** Through 10-fold cross-validation, the model showed an average accuracy of **96.77%**. This method provides a more robust estimate of the model's performance on unseen data by splitting the training data into multiple folds and training/testing on different combinations. This helps to reduce bias and variance in the accuracy estimation.
    * **Test Accuracy:** On the independent test set, the model achieved an accuracy of **97.14%**. This score reflects how well the trained model generalizes to new, unobserved data.

* **Confusion Matrix Analysis:**
    The confusion matrix provides a detailed breakdown of the model's predictions:

    ```
    [[83  2]
     [ 2 53]]
    ```

    From this matrix, we can interpret:
    * **True Negatives (TN):** 84 instances were correctly predicted as benign (non-cancerous).
    * **False Positives (FP):** 3 instances were incorrectly predicted as malignant (cancerous) when they were actually benign. This is also known as a **Type I error**.
    * **False Negatives (FN):** 3 instances were incorrectly predicted as benign (non-cancerous) when they were actually malignant. This is also known as a **Type II error**, and in medical contexts like cancer detection, this can be particularly critical.
    * **True Positives (TP):** 47 instances were correctly predicted as malignant (cancerous).

    **Understanding Accuracy from the Confusion Matrix:**
    Accuracy is calculated as: $$(TP + TN) / (TP + TN + FP + FN)$$
    For our model, this is: $$(53 + 83) / (53 + 83 + 2 + 2) = 136 / 140 \approx 0.9714 $$ or **97.14%**.

These results collectively highlight the model's high predictive power and its capability to accurately distinguish between benign and malignant cases, minimizing critical errors where possible.

---
---

## Dataset

The project utilizes the **Breast Cancer Wisconsin (Original) Dataset**, obtained from the UCI Machine Learning Repository. This dataset contains various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, used to predict whether a tumor is benign or malignant.

You can find more information about the dataset [here](https://www.kaggle.com/datasets/mariolisboa/breast-cancer-wisconsin-original-data-set). Please ensure the `breast_cancer_bd.csv` file in your project directory is derived from this dataset.

---
## Contributing

Feel free to fork this repository, suggest improvements, or open issues. Any contributions are welcome!

---

