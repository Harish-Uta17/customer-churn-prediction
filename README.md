# 🔮 ChurnGuard AI | Customer Retention Intelligence

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/AI-Scikit%20Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

ChurnGuard AI is an advanced machine learning application designed to predict customer churn in the telecommunications sector.  
It analyzes customer demographics, service usage, and billing patterns to identify at-risk customers and help businesses take proactive actions.

---

## 🚀 Overview

Customer churn is one of the biggest challenges for subscription-based businesses.  
This project helps organizations to:

- Predict whether a customer is likely to churn  
- Measure churn probability  
- Categorize customers based on risk level  
- Make data-driven retention decisions  

---

## 🔑 Key Features

- 🤖 Predictive AI Engine using Machine Learning  
- 📊 Interactive Dashboard built with Streamlit  
- 📉 Churn Risk Quantification  
- 💡 Actionable Business Insights  
- ⚡ Real-Time Prediction System  

---

## 🛠️ Tech Stack

- Programming Language: Python  
- Frontend: Streamlit  
- Data Processing: Pandas, NumPy  
- Machine Learning: Scikit-Learn, XGBoost, LightGBM  
- Imbalanced Data Handling: SMOTE  
- Visualization: Matplotlib, Plotly  

---

## 📂 Project Structure

customer-churn-prediction/  
├── data/  
│   ├── raw/                # Original dataset  
│   └── processed/          # Cleaned and engineered data  
│  
├── models/  
│   ├── best_model.pkl      # Trained ML model  
│   └── feature_names.pkl   # Feature list  
│  
├── src/  
│   ├── data_loader.py      # Load dataset  
│   ├── feature_engineer.py # Feature engineering  
│   ├── model_trainer.py    # Training pipeline  
│   ├── predictor.py        # Prediction pipeline  
│   └── config.py           # Project configuration  
│  
├── dashboard.py            # Streamlit application  
├── requirements.txt        # Dependencies  
└── README.md               # Documentation  

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

git clone https://github.com/your-username/customer-churn-prediction.git  
cd customer-churn-prediction  

---

### 2. Create a Virtual Environment

Windows:

python -m venv venv  
venv\Scripts\activate  

Mac/Linux:

python3 -m venv venv  
source venv/bin/activate  

---

### 3. Install Dependencies

pip install -r requirements.txt  

---

### 4. Train the Model

Before running predictions, train the model:

python -m src.model_trainer  

This will:

- Load dataset  
- Perform feature engineering  
- Train multiple ML models  
- Save the best model automatically  

---

### 5. Run Prediction Script

To test predictions from terminal:

python -m src.predictor  

You will be asked to enter customer details and the system will return:

- Churn probability  
- Risk level  
- Final prediction  

---

### 6. Run Streamlit Dashboard

streamlit run dashboard.py  

The application will open in your browser at:

http://localhost:8501  

---

## 🧠 Model Performance

The model was trained on the Telco Customer Churn dataset.

### Performance Metrics

- ROC-AUC Score: ~0.84  
- Accuracy: ~80–85%  
- Recall (Churn Class): High sensitivity to detecting churners  

### Models Compared

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  

The system automatically selects the best performing model and saves it for future predictions.

---

## 🎯 How the System Works

1. User provides customer details  
2. System applies preprocessing and feature engineering  
3. Trained ML model predicts churn probability  
4. Risk level is calculated  

Risk Levels:

- Probability > 70%  → High Risk  
- Probability 40–70% → Medium Risk  
- Probability < 40%  → Low Risk  

---

## 🔍 Prediction Output Example

When running python -m src.predictor, the system asks for input like:

Gender: Male  
Senior Citizen: 0  
Partner: Yes  
Tenure: 12  
Monthly Charges: 65.5  
Total Charges: 780  

And returns:

Will Churn: True  
Churn Probability: 82.45%  
Risk Level: High  

---

## 🚀 Future Enhancements

Planned improvements:

- Database integration  
- REST API using Flask/FastAPI  
- Batch prediction support  
- Cloud deployment  
- Advanced analytics dashboard  

---

## 🤝 Contributing

Contributions are welcome!

- Fork the repository  
- Create a new feature branch  
- Submit a pull request  

---

## 📧 Contact

Your Name  

LinkedIn: https://linkedin.com/in/your-profile  
Portfolio: https://your-portfolio.com  
Email: your.email@example.com  

---

### ⭐ If you like this project, give it a star on GitHub!
