# Customer-Churn-Prediction


**Customer Churn Prediction**

**Project Overview**

This project is designed to help predict whether a customer of a telecom company is likely to churn (i.e., leave the service). Customer churn is a significant issue in the telecom industry, as it directly impacts revenue and profitability. By using machine learning models, we aim to predict the likelihood of churn based on various customer features and account details.

The project goes through:

- Extensive Exploratory Data Analysis (EDA) to uncover insights and patterns
-Data preprocessing to clean and transform the data
-Feature engineering to improve model performance
-Model building with multiple machine learning algorithms
-Class imbalance handling with advanced techniques like SMOTEENN
-Model evaluation using various performance metrics
-Deployment of a Flask web application where users can input customer details to predict churn.

This project is highly useful for business teams to:
-Identify customers who are at high risk of churning
-Create retention strategies and personalized offers
-Improve customer service to reduce churn rates

Table of Contents
-Project Overview
-Business Problem
-Dataset
-Features
-Requirements
-Installation
-Data Preprocessing
-Exploratory Data Analysis (EDA)
-Feature Engineering
-Model Building and Evaluation
-Handling Imbalanced Classes
-Web App Deployment
-Results
-Future Work
-How to Run
-Contributing
-License

**Business Problem**
Churn prediction is a critical problem faced by telecom companies. Losing customers has a direct impact on profitability, and acquiring new customers is often more expensive than retaining existing ones. This project attempts to solve the problem of predicting which customers are likely to churn, based on their demographic details, subscription plans, and usage patterns.

A well-performing churn prediction model can help the business:
-Reduce churn rates through proactive customer retention
-Target at-risk customers with tailored offers
-Improve overall customer satisfaction and loyalty

**Dataset**
The dataset used in this project is obtained from a telecom company and contains various details about customers, their accounts, and the services they use. The key aspects of the dataset are:

Rows: Each row represents a customer.
Columns: Each column provides information about the customer (demographics, subscription details, etc.).
Key features include:

Demographic details: Gender, Age, Senior Citizen status, etc.
Subscription details: Phone Service, Internet Service, Contract Type, etc.
Account details: Tenure, Monthly Charges, Total Charges, Payment Method, etc.
Target variable: Churn (Yes or No)
This dataset contains both categorical and numerical variables, and we handle missing values, outliers, and feature transformations in the data preprocessing phase.

**Features**
Here is a list of the key features in the dataset:

SeniorCitizen: Whether the customer is a senior citizen.
MonthlyCharges: The amount charged to the customer on a monthly basis.
TotalCharges: The total amount charged during the customer's tenure.
Gender: Gender of the customer.
Partner: Whether the customer has a partner.
Dependents: Whether the customer has dependents.
PhoneService: Whether the customer has phone service.
MultipleLines: Whether the customer has multiple lines.
InternetService: Type of internet service used (DSL, Fiber, or None).
Contract: Type of contract (Month-to-Month, One Year, Two Years).
Tenure: Duration of the customer's subscription (in months).

**Requirements**
To replicate this project, you will need the following dependencies installed in your Python environment:
Python 3.x
Flask: For deploying the web application.
scikit-learn: For machine learning algorithms and model evaluation.
imbalanced-learn: For handling class imbalance with SMOTEENN.
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib and seaborn: For data visualization.
Bootstrap (optional): For styling the web app front end.
To install the required dependencies, run:

bash
Copy code
pip install -r requirements.txt

**Installation**
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/customer-churn-prediction.git
Navigate to the project folder:
bash
Copy code
cd customer-churn-prediction
Install the dependencies:
bash
Copy code
pip install -r requirements.txt

**Data Preprocessing**
The data preprocessing phase involves:

Handling missing values in key columns.
Converting categorical variables to numerical representations using One-Hot Encoding.
Scaling numerical features such as MonthlyCharges and TotalCharges for better model performance.
Handling outliers in the dataset.
Class imbalance in the dataset (more "No Churn" customers than "Yes Churn" customers) is addressed using SMOTEENN, a combination of SMOTE (Synthetic Minority Oversampling Technique) and ENN (Edited Nearest Neighbors).

**Exploratory Data Analysis (EDA)**
EDA is performed to understand data distributions, relationships between variables, and customer churn patterns. We use various visualizations, including:

Histograms and boxplots for numerical features.
Bar plots for categorical features.
Heatmaps to check correlations between different features.
Key insights include:

Senior citizens are more likely to churn.
Customers with month-to-month contracts are at higher risk of churn.
High monthly charges contribute to customer churn.

**Feature Engineering**
In the feature engineering phase, we create new features and modify existing ones to enhance model performance. Some techniques used include:

Combining related features (e.g., multiple services) into a single feature.
Creating interaction features to capture relationships between variables.
Removing unnecessary or redundant features.

**Model Building and Evaluation**
We experiment with multiple machine learning algorithms to predict customer churn:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier

**Evaluation Metrics:**
Accuracy: Measures overall correctness of the model.
Precision: Measures how many of the predicted positives are actually positives.
Recall: Measures how many actual positives were correctly identified by the model.
F1 Score: The harmonic mean of precision and recall.
Confusion Matrix: To visualize true positives, false positives, true negatives, and false negatives.
After tuning hyperparameters, Random Forest Classifier performs the best, with SMOTEENN significantly improving the recall for the minority class.

**Handling Imbalanced Classes**
The dataset exhibits class imbalance, with a higher number of "No Churn" cases compared to "Yes Churn" cases. To address this, we use:

SMOTE: Generates synthetic samples for the minority class.
ENN: Removes noisy data points from the majority class.
By applying SMOTEENN, we can improve the model's ability to correctly predict customers who are likely to churn.

**Web App Deployment**
The project includes a web-based application built using Flask. This app allows users to input customer information (e.g., Senior Citizen, Monthly Charges, Contract Type, etc.) and predict whether the customer is likely to churn.

To run the app:

Run the Flask server:
bash
Copy code
python app.py
Access the app in your browser at http://127.0.0.1:5000.

**Results**
The Random Forest model with SMOTEENN provides the best results:

Accuracy: 79%
Precision for Churn class: 69%
Recall for Churn class: 50%
F1 Score for Churn class: 58%
These results demonstrate that the model is reasonably effective at predicting customer churn, especially after handling class imbalance.

**Future Work**
Potential improvements for future versions of the project include:

Advanced Feature Engineering: Creating more complex interaction features or using domain knowledge for feature selection.
Deep Learning Models: Testing deep learning models (e.g., neural networks) to see if they improve performance.
Customer Segmentation: Implement clustering to segment customers and create personalized retention strategies.
Real-time Prediction: Deploy the model in a production environment with real-time churn prediction capabilities.

**How to Run**
Running the Jupyter Notebook
Launch the notebook:
bash
Copy code
jupyter notebook
Open the Churn Analysis EDA & ModelBuilding.ipynb file and execute the cells to run the analysis and model building.
Running the Flask Web Application
Navigate to the project directory.
Start the Flask app:
bash
Copy code
python app.py
Go to http://127.0.0.1:5000 in your web browser.



