README.md: Provides an overview of the project, including the problem statement, approach, and results.
data/: Contains the dataset used for training and testing the model.
customer_churn.csv
notebooks/: Includes the Jupyter notebooks used for data exploration, model training, and evaluation.
01_data_exploration.ipynb
02_model_training.ipynb
03_model_evaluation.ipynb
models/: Stores the trained models and any saved checkpoints.
customer_churn_ann_model.h5
scripts/: Contains Python scripts used for data preprocessing, model training, and prediction.
data_preprocessing.py
train_model.py
predict_churn.py
results/: Stores the output of the model, including confusion matrices, accuracy scores, and other metrics.
confusion_matrix.png
evaluation_metrics.txt
requirements.txt: Lists the Python libraries and dependencies needed to run the project.
.gitignore: Specifies files and directories to be ignored by Git.
Key Features:
Data Exploration: Analyze customer attributes, understand patterns, and visualize key insights.
Model Building: Implementation of an ANN model using TensorFlow/Keras to predict customer churn.
Model Evaluation: Confusion matrix, accuracy, precision, recall, and F1-score are calculated to evaluate model performance.
Visualization: Includes visual representations of the confusion matrix and other important metrics.
Usage:
Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Run the Jupyter notebooks or scripts to explore the data, train the model, and evaluate the results.
Results:
The ANN model achieved a 79% accuracy in predicting customer churn. The precision for class 0 (customers who did not churn) was 83%, and for class 1 (customers who actually churned), it was 65%.

License:
This project is licensed under the MIT License.
