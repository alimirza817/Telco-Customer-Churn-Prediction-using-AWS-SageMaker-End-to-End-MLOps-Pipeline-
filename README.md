# Telco-Customer-Churn-Prediction-using-AWS-SageMaker-End-to-End-MLOps-Pipeline-
## 📖 Overview

This project presents a production-grade end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. The system leverages AWS SageMaker, XGBoost, and MLOps principles to automate the entire workflow from data preprocessing to batch inference. The goal is to identify customers likely to churn so that businesses can take proactive retention actions.

## 🎯 Problem Statement

Customer churn prediction is a highly imbalanced classification problem, where:

*  73.5% customers → No churn
*  26.5% customers → Churn

A naive model would achieve high accuracy but fail to detect churners.

👉 Focus: Maximize Recall & F1-score instead of accuracy.

## 📊 Dataset
1. Source: IBM Telco Customer Churn Dataset
2. Records: 7,043 customers
3. Features: 20+ attributes
    *  Demographics
    *  Account details
    *  Services used
    *  Link-> https://www.kaggle.com/datasets/blastchar/telco-customer-churn 
## 🔍 Exploratory Data Analysis (EDA)
Key insights:
1. Month-to-month contracts → highest churn (~43%)
2. Electronic check users → highest churn (~45%)
3. High monthly charges → more churn
4. Low tenure customers → highest risk
## ⚙️ Feature Engineering
✅ Data Cleaning
*  Handled missing values in TotalCharges
*  Removed customerID
  
✅ Engineered Features
*  AvgMonthlySpend
*  TenureBucket
*  ServiceCount
*  HighValue
*  RiskCombo
*  NoInternetServices
  
✅ Encoding
*  Label Encoding (binary features)
*  Ordinal Encoding (contract)
*  One-Hot Encoding (categorical)
  
✅ Scaling
*  StandardScaler applied to numeric features
## ⚖️ Handling Class Imbalance

Three-layer strategy:

1. SMOTE (oversampling minority class)
2. scale_pos_weight (XGBoost)
3. Stratified Splitting
## 🧠 Model Training
*  Algorithm: XGBoost (SageMaker Built-in)
*  Framework: AWS SageMaker
*  Optimization: Bayesian Hyperparameter Tuning
## 📈 Model Performance
✅ Final Test Results:
*  ROC AUC: 0.8507
*  Accuracy: 75.6%
*  Recall: 77.1%
*  Precision: 52.7%
*  F1 Score: 0.63
## 🎯 Threshold Optimization
*  Default threshold (0.5) → poor precision
*  Optimized threshold → 0.737

👉 Improved F1-score significantly

## ⚡ Batch Inference
1. Used SageMaker Batch Transform

2. Generated churn predictions for all customers

3. Risk segmentation:
   *  Low
   *  Medium
   *  High
   *  Very High
## 🔁 MLOps Pipeline (SageMaker)
The pipeline automates the workflow:
1. Data Processing
2. Model Training
3. Model Registration
4. Batch Inference

## 🏗️ Project Architecture
<img width="1204" height="482" alt="image" src="https://github.com/user-attachments/assets/d94477aa-a19c-4122-bd82-a323e867be94" />

## 📂 Repository Structure
Telco_Customer_Churn_Prediction Using Sagemaker

├── Data/

│ └── train.csv

│ └── validation.csv

│ └── test.csv

├── Notebooks/

│ └── EDA_Feature_Engineering.ipynb

│ └── SageMaker_Training.ipynb

├── Pipeline/

│ └── Customer_Churn.py

├── README.md

└── requirements.txt

## 🛠️ Technologies Used
*  Python
*  Pandas, NumPy, Scikit-learn
*  XGBoost
*  AWS SageMaker
*  AWS S3
*  AWS Lambda
*  AWS CloudWatch

## 💼 Business Impact
1. Identifies 77% of churners
2. Enables targeted retention campaigns
3. Reduces revenue loss significantly

## 📸 Screenshots
### 📊 Exploratory Data Analysis (EDA)
<img width="1163" height="454" alt="image" src="https://github.com/user-attachments/assets/189026f3-f455-4745-a4f9-019ddf479776" />

<img width="1097" height="480" alt="image" src="https://github.com/user-attachments/assets/ef3bbf4f-7b8a-4238-8f69-b54bb44e3673" />

#### Correlation Heatmap
<img width="804" height="493" alt="image" src="https://github.com/user-attachments/assets/d3f4be0b-e040-4824-891b-0aaf83f89ad2" />

### ⚙️ Feature Engineering
<img width="468" height="123" alt="image" src="https://github.com/user-attachments/assets/3e0ba815-127c-44d8-8ba4-ae70ec0362c5" />

### ☁️ SageMaker Training Job
<img width="718" height="143" alt="image" src="https://github.com/user-attachments/assets/18782ab8-59cb-4a30-a3a1-9cd3baad35ce" />

### 🔍 Hyperparameter Tuning
<img width="1104" height="209" alt="image" src="https://github.com/user-attachments/assets/b5b91d04-9956-4eb2-895e-9582b5e7b0d6" />

### 📈 Model Evaluation
<img width="613" height="484" alt="image" src="https://github.com/user-attachments/assets/f4d30779-f7e8-4ded-b37a-5b2c85ee740a" />

<img width="713" height="481" alt="image" src="https://github.com/user-attachments/assets/8937cd40-e946-4955-aca2-5c640224b0e0" />

<img width="835" height="481" alt="image" src="https://github.com/user-attachments/assets/be607cda-3641-4123-a1fc-9249e593158d" />

### 🔁 SageMaker Pipeline
<img width="1181" height="519" alt="image" src="https://github.com/user-attachments/assets/ff095e7c-f9fb-47b2-b202-db8a55cc5d1f" />

## 🚀 How to Run
### 1️⃣ Clone the Repository
  *  git clone https://github.com/alimirza817/Telco-Customer-Churn-Prediction-using-AWS-SageMaker-End-to-End-MLOps-Pipeline-.git
  *  cd Telco-Customer-Churn-Prediction-using-AWS-SageMaker-End-to-End-MLOps-Pipeline-
### 2️⃣ Create Virtual Environment (Recommended)
  *  python -m venv venv
  *  source venv/bin/activate      # Mac/Linux
  *  venv\Scripts\activate         # Windows
### 3️⃣ Install Dependencies
  *  pip install -r requirements.txt # Install those to latest versions
### 4️⃣ Run EDA & Feature Engineering

Open Jupyter Notebook:

jupyter notebook

Run:

*  EDA_Feature_Engineering.ipynb

This step will:

  *  Clean the dataset
  *  Perform feature engineering
  *  Generate train, validation, and test datasets
### 5️⃣ Upload Data to AWS S3
*  Create an S3 bucket
*  Upload:
    *  train.csv
    *  validation.csv
    *  test.csv
### 6️⃣ Configure AWS Credentials

Make sure AWS CLI is configured:

*  aws configure

Provide:

*  Access Key
*  Secret Key
*  Region (e.g., us-east-1)
### 7️⃣ Run SageMaker Training

Open notebook:

*  SageMaker_Training.ipynb

This will:

*  Train XGBoost model on SageMaker
*  Perform hyperparameter tuning
*  Save model artifact to S3
### 8️⃣ Execute SageMaker Pipeline
*  Run pipeline script or notebook
*  Steps executed:
    *  Data Processing
    *  Model Training
    *  Model Registration
    *  Batch Inference
### 9️⃣ Run Batch Inference
*  Use SageMaker Batch Transform
*  Input: test dataset from S3
*  Output: predictions stored in S3
### 🔟 View Results
*  Download predictions from S3
*  Open predictions.csv
*  Analyze churn probabilities and risk categories
### ✅ Output
*  Trained XGBoost model
*  Evaluation metrics (AUC, F1-score)
*  Batch predictions
*  Risk segmentation of customers
### ⚠️ Notes
*  AWS costs may apply for SageMaker usage
*  Ensure IAM roles have proper permissions
*  Use smaller instance types for testing






