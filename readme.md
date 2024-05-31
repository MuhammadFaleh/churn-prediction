# Comparative Analysis of Traditional, Hybrid, and Deep Learning Algorithms for Predicting Telecom Customer Churn

## Table of Contents
- [Introduction](#introduction)
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

## Introduction
Customer churn, defined as customers transitioning from one service provider to another, presents a significant challenge across various industries, particularly within telecommunications. This phenomenon directly impacts both revenue streams and the competitive landscape. Thus, many businesses are increasingly acknowledging the necessity of addressing this issue to prevent potential profit declines stemming from customer churn.

## Abstract
Over recent years, the telecommunications sector has sought to implement predictive methods for customer churn, thereby retaining existing customers while attracting new ones. This study aims to develop and compare machine learning and deep learning techniques for identifying customer churn within the telecommunications industry through customer behavior analysis.

We propose an extensive experimental and comparative investigation using traditional and ensemble machine learning algorithms (Random Forests, XGBoost, Decision Trees, and Multilayer Perceptron), deep learning algorithms (Convolutional Neural Networks, Deep Recurrent Neural Networks), and hybrid models that integrate these methods. These models' performance is compared using key evaluation metrics such as accuracy, precision, F1 score, AUC, and ROC. Preliminary results indicate that hybrid models, inclusive of deep learning algorithms, show promise in outperforming traditional and ensemble machine learning algorithms.

**Keywords:** Churn Prediction, Traditional Algorithms, Ensemble Algorithms, Deep Learning Algorithms, Hybrid Algorithms.

## Dataset
The dataset used for this project contains customer behavior data from a telecommunications company. The dataset includes various features such as customer demographics, service usage patterns, and contract details.

### Dataset Description
The dataset is obtained from a telecommunications provider and contains data on around 100,000 consumers. Key features include traffic type, traffic destination, rate plan, customer loyalty, and traffic behavior, divided into sub-datasets of traffic data and customer profile factors. The target variable is the customer's status (ACTIVE, CHURN) four months after the month in which traffic occurred.

## Methodology
Our approach involves the following steps:
1. **Data Preprocessing:** Cleaning and preparing the data for modeling, including handling missing data, feature engineering, and balancing the dataset using SMOTE.
2. **Feature Engineering:** Creating new features to improve model performance, such as customer tenure and encoded categorical features.
3. **Model Development:** Implementing various machine learning and deep learning models.
4. **Model Evaluation:** Comparing model performance using key metrics such as accuracy, precision, F1 score, AUC, and ROC.

### Models Implemented:
- **Traditional and Ensemble Machine Learning Algorithms:**
  - Random Forests
  - XGBoost
  - Decision Trees
  - Multilayer Perceptron
- **Deep Learning Algorithms:**
  - Convolutional Neural Networks (CNN)
  - Deep Recurrent Neural Networks (DRNN)
- **Hybrid Models:**
  - Integrating both machine learning and deep learning techniques.

## Installation
To run this project locally, please follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/churn-prediction.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure you have Jupyter Notebook installed to run the provided notebook.

## Usage
To use this project, open the `churn_prediction_analysis.ipynb` notebook in Jupyter Notebook and follow the steps outlined in the notebook to preprocess the data, train the models, and evaluate their performance.

## Results
The results of our study indicate that hybrid models, which include deep learning algorithms, tend to outperform traditional and ensemble machine learning algorithms in predicting customer churn. Detailed results, including performance metrics and visualizations, can be found in the `Results` section of the provided notebook.

### Key Findings:
- Hybrid models combining ensemble boosting and deep learning exhibited the highest performance with an accuracy of 97.70%, precision of 97.90%, recall of 97.49%, and an F1-score of 97.67%.
- Traditional algorithms, such as Decision Trees and Multilayer Perceptron, showed lower performance compared to ensemble and hybrid models.
- SMOTE balancing significantly improved the performance metrics for recall, precision, and F1-score.

## Conclusion
Our study demonstrates the effectiveness of hybrid models in predicting customer churn within the telecommunications industry. These models leverage the strengths of both traditional machine learning and deep learning techniques, providing a robust solution for churn prediction.

## Future Work
Future work may include:
- Exploring additional features that can improve model performance.
- Investigating the impact of different data preprocessing techniques.
- Implementing real-time churn prediction systems.
- Extending the study to other industries beyond telecommunications.
- Exploring the use of transfer learning and domain adaptation techniques.

## Contributors
- **Amir AL-Maamari** 
- **Salem Almutiri** 
- **Muhammad Almutiri** 
- **Supervisor:** Dr. Ali Mustafa Qamar

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
