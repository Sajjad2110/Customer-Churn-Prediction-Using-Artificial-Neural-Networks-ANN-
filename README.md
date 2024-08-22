
# Customer Churn Prediction Using Artificial Neural Network (ANN)

This repository contains the code, data, and results for a project focused on predicting customer churn using an Artificial Neural Network (ANN). The goal of this project is to identify customers who are likely to leave a service provider, enabling the company to take proactive measures to retain them. The project uses a dataset from IBM Sample Data Sets, which includes various customer attributes, such as their services, account information, and demographics.

## Problem Statement

Customer churn is a critical issue for service providers, as retaining existing customers is more cost-effective than acquiring new ones. The objective of this project is to build a predictive model using ANN to identify customers who are likely to churn. This allows the company to implement retention strategies, thereby reducing the churn rate.

## Dataset

The dataset used in this project is provided by IBM and includes information on:

- **Churn Status**: Whether the customer left within the last month.
- **Services Signed Up**: Phone, multiple lines, internet, online security, online backup, device protection, tech support, streaming TV, and movies.
- **Customer Account Information**: Duration of being a customer, contract type, payment method, paperless billing, monthly charges, and total charges.
- **Demographics**: Gender, age range, and whether the customer has partners and dependents.

## Project Workflow

1. **Data Preprocessing**: 
   - Handled missing values, encoded categorical variables, and scaled numerical features.
   - Split the data into training and testing sets.

2. **Model Development**:
   - Developed an Artificial Neural Network (ANN) model using TensorFlow and Keras.
   - The model consists of input, hidden, and output layers designed to predict customer churn.

3. **Model Evaluation**:
   - Evaluated the model using accuracy, precision, recall, and confusion matrix.
   - The model achieved an accuracy of 79%, with precision scores for non-churning and churning customers at 83% and 65%, respectively.

4. **Results Visualization**:
   - Visualized the performance of the model using a confusion matrix.

## Results

The model provides the following results:
- **Accuracy**: 79%
- **Precision (Class 0 - Not Churned)**: 83%
- **Precision (Class 1 - Churned)**: 65%

These results demonstrate the model's capability to correctly identify customers who are likely to churn, with an overall balanced performance.

## Conclusion

This project showcases the application of an Artificial Neural Network to predict customer churn. The insights gained from this model can be leveraged by companies to target at-risk customers with retention campaigns, thereby reducing churn and increasing customer lifetime value.

## How to Use

1. Clone the repository.
2. Ensure you have Python and the necessary libraries installed (e.g., TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn).
3. Run the Jupyter notebook to train the model and visualize the results.

## Repository Structure

- **CustomerChurnPredictionUsingANN.ipynb**: The Jupyter notebook containing the code and analysis.
- **data/**: Directory containing the dataset (if provided).
- **images/**: Directory containing images related to the project (e.g., confusion matrix).

## Acknowledgments

- Dataset: [IBM Sample Data Sets](https://www.ibm.com/analytics/datasets)
- Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn
