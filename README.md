# 🛡️ Credit Card Fraud Detection

A complete end-to-end machine learning pipeline for detecting fraudulent credit card transactions using multiple classification models.  
This project covers data exploration, preprocessing, class imbalance handling, model training, evaluation with appropriate metrics, and decision threshold analysis to align technical performance with real-world business objectives.

---

## 🧠 Business Context

Credit card fraud detection is a high-stakes problem where different types of errors have different business costs:

- **False Negatives (missed fraud):** lead to direct financial loss and chargebacks.  
- **False Positives (false alarms):** cause customer friction and may impact customer trust.

Because missing fraud is typically more costly than incorrectly flagging legitimate transactions, this project prioritizes **Recall** while monitoring **Precision** to keep false alarms at an acceptable level.  
The final model is designed as a **decision-support system** by enabling adjustment of the **decision threshold** based on organizational risk tolerance.

---

## 📁 Project Structure

├── credit_card_fraud_detection.ipynb # Main notebook
├── README.md # Project documentation

---

## 📊 Dataset Overview

This dataset contains anonymized credit card transaction records with 30 features:

- **Time** — Seconds since the first transaction  
- **Amount** — Transaction amount  
- **V1–V28** — PCA-transformed components  
- **Class** — Target variable (0 = normal, 1 = fraud)

📌 The dataset is **highly imbalanced**, with only **0.17%** fraud cases.

---

## 🔍 Exploratory Data Analysis (EDA)

Includes:

- Class distribution  
- Amount distribution  
- Time distribution  
- Correlation heatmap  

Key insights are documented within the notebook.

---

## ⚙️ Data Preprocessing

- Splitting features (X) and labels (y)  
- Standard scaling applied **only to `Amount`**  
- Stratified train-test split  
- SMOTE applied **only on training data**  
- Ensures **no data leakage**

---

## 🤖 Machine Learning Models

### 1️⃣ Logistic Regression
- Very high recall  
- **Extremely low precision**  
- ❌ Not suitable for production due to a high number of false alarms

### 2️⃣ Random Forest
- Best overall balance  
- Strong F1-score  
- Reliable and stable performance

### 3️⃣ XGBoost
- Excellent recall  
- Best for minimizing missed fraud cases in high-risk scenarios

---

## 📈 Model Performance Comparison

| Model                | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---------------------|-------------------|----------------|----------|
| Logistic Regression | 0.12              | 0.90           | 0.21     |
| Random Forest       | 0.82              | 0.82           | 0.82     |
| XGBoost             | 0.76              | 0.86           | 0.81     |

🔥 **Best overall:** Random Forest  
🔥 **Best at detecting fraud:** XGBoost

---

## 📉 ROC Curve Analysis

ROC curves are included for:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Combined ROC comparison  

---

## ⭐ Feature Importance

Feature importance is visualized for:

- Random Forest  
- XGBoost  

This helps identify which features most influence fraud detection.

---

## 🎯 Decision Threshold Analysis

Rather than using the default classification threshold (0.5), multiple thresholds were evaluated to understand the trade-off between **Recall** and **Precision**.

- Lower thresholds increase Recall (catch more fraud) at the cost of more false positives.  
- Higher thresholds increase Precision (fewer false alarms) but may miss more fraud cases.

Given the business objective of minimizing undetected fraud, a **lower operating threshold** was selected to prioritize Recall while maintaining an acceptable level of Precision.  
This transforms the model from a static classifier into a **decision-support tool** that can be tuned based on risk tolerance.

---

## 🔚 Final Conclusion

This project demonstrates how machine learning can be applied to a real-world, high-risk financial problem.  
Multiple models were evaluated using metrics beyond accuracy, with a focus on **Precision, Recall, and F1-score** due to severe class imbalance.

By incorporating **decision threshold analysis**, the system moves beyond simple prediction and supports business decision-making.  
A lower threshold was selected to prioritize Recall, minimizing the risk of undetected fraudulent transactions in line with financial risk objectives.

Overall:
- **Random Forest** provides the best balance between precision and recall.  
- **XGBoost** achieves the highest recall, making it suitable when missing fraud is extremely costly.  
- **Logistic Regression** is not suitable for deployment due to very low precision.

This approach highlights how model performance can be aligned with real business priorities, balancing fraud prevention and customer experience.

---

## 🔮 Future Work

- Hyperparameter tuning  
- Testing LightGBM and CatBoost  
- Anomaly detection (Isolation Forest, Autoencoders)  
- Cost-sensitive learning  
- Real-time fraud detection system

---

## 🧩 Requirements

numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
xgboost

---

## 👩‍💻 Author

**Abeer Monir Barakat**  
AI Engineer  
GitHub: https://github.com/abeermonir90
