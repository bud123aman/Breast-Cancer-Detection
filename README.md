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

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bud123aman/Breast-Cancer-Detection.git](https://github.com/bud123aman/Breast-Cancer-Detection.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Breast-Cancer-Detection
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn
    ```
4.  **Ensure you have the `breast_cancer.csv` dataset** in the project directory.

---

## Usage

You can run the Jupyter Notebook directly or execute the Python code cells:

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Breast_Cancer_Detection.ipynb
    ```
    Then, run all the cells in the notebook.

2.  **Alternatively, run the Python script (if converted):**
    If you've converted the notebook to a Python script (e.g., `main.py`), you can run it using:
    ```bash
    python main.py
    ```

The output will display the confusion matrix, the accuracy calculated using K-Fold Cross-Validation, and the test accuracy.

---

## Model Performance

The model demonstrates strong performance in detecting breast cancer:

* **Accuracy (K-Fold Cross-Validation):** The model achieved an average accuracy of **96.70%** across 10 folds, indicating good generalization capabilities.
* **Test Accuracy:** On the held-out test set, the model achieved an accuracy of **95.62%**.
* **Confusion Matrix:**
    ```
    [[84  3]
     [ 3 47]]
    ```
    This matrix shows:
    * **84** true negatives (correctly predicted as benign)
    * **47** true positives (correctly predicted as malignant)
    * **3** false positives (incorrectly predicted as malignant)
    * **3** false negatives (incorrectly predicted as benign)

These results highlight the model's ability to accurately distinguish between benign and malignant cases with a low number of misclassifications.

---

## Dataset

The project utilizes the **Breast Cancer Wisconsin (Original) Dataset**, obtained from the UCI Machine Learning Repository. This dataset contains various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, used to predict whether a tumor is benign or malignant.

You can find more information about the dataset [here](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). Please ensure the `breast_cancer.csv` file in your project directory is derived from this dataset.

---
## Contributing

Feel free to fork this repository, suggest improvements, or open issues. Any contributions are welcome!

---

