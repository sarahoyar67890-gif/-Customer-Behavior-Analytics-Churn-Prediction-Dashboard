# 🔮 Customer Churn Prediction Dashboard

> An end-to-end Machine Learning system that analyzes customer behavior, generates business insights, and predicts churn probability with 80%+ accuracy.

---

## 👩‍💻 About This Project

| | |
|---|---|
| **Author** | Sara Ahmad |
| **Task ID** | ML-INT-1 |
| **Internship** | Teyzix Core — June Batch 2026 |
| **Domain** | Machine Learning |
| **Deadline** | 19 June 2026 |

---

## 📌 Problem Statement

Telecommunications companies lose millions in revenue every year when customers leave without warning. This project builds a complete ML system that identifies **which customers are at risk of churning — before they leave** — giving businesses time to act.

---

## 📁 Project Structure

```
ML-INT-1_Sara_Ahmad/
│
├── 📓 customer_churn.ipynb         — Main Jupyter Notebook
├── 🐍 customer_churn.py            — Python Script
├── 🖥️  streamlit_app.py            — Interactive Dashboard
├── 📊 WA_Fn-UseC_-Telco-Customer-Churn.csv  — Dataset
├── 📄 Model_Results_Report.docx    — Model Performance Report
├── 📄 Summary_Insights_Document.docx — Business Insights
├── 📄 Self_Assessment.docx         — Self Assessment
└── 📖 README.md                    — You are here
```

---

## 🚀 How to Run

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib shap streamlit
```

### Run the notebook
```bash
jupyter notebook customer_churn.ipynb
```

### Run the Streamlit dashboard
```bash
streamlit run streamlit_app.py
```

---

## 🛠️ What This Project Does

### 1. 📊 Data Analysis
- 7,043 customer records loaded and explored
- Missing values handled with median imputation
- Outlier analysis with boxplots

### 2. ⚙️ Feature Engineering — 8 Behavioral Features

| Feature | Description | Type |
|---|---|---|
| `AvgSpend` | TotalCharges / (tenure + 1) | Trend-based |
| `ServicesCount` | Number of active services | Usage frequency |
| `LongTermCustomer` | Tenure > 24 months flag | Tenure pattern |
| `PaymentRisk` | Electronic check flag | Payment history |
| `SupportRisk` | No tech support flag | Support interaction |
| `ContractRisk` | Contract type risk score (0–2) | Behavioral |
| `ChargesPerService` | Monthly charges per service | Trend-based |
| `LoyaltyRisk` | Long tenure + high charges flag | Behavioral |

### 3. 📈 Data Visualization
- Churn distribution
- Revenue trend by tenure
- Contract type vs churn
- Internet service vs churn
- Payment method vs churn
- Monthly charges distribution by churn
- Services count vs churn
- Correlation heatmap

### 4. 🤖 Machine Learning Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.98% | 67.79% | 54.01% | 60.12% | 0.8474 |
| Random Forest | 79.56% | 64.05% | 52.41% | 57.65% | 0.8351 |
| Gradient Boosting | 78.99% | 62.42% | 52.41% | 56.98% | 0.8343 |

### 5. 🧩 Customer Segmentation

| Segment | Customers | Percentage |
|---|---|---|
| 🔴 High Value | 1,259 | 17.9% |
| 🟡 Medium Value | 1,611 | 22.9% |
| 🟢 Low Value | 4,173 | 59.3% |

### 6. 🎯 Risk Classification

| Risk Level | Customers | Threshold |
|---|---|---|
| 🔴 High Risk | 704 | Probability > 70% |
| 🟡 Medium Risk | 1,884 | 30% – 70% |
| 🟢 Low Risk | 4,455 | Probability < 30% |

---

## ⭐ Bonus Features

| Feature | Status |
|---|---|
| Streamlit Dashboard | ✅ Complete |
| SHAP Explainability | ✅ Complete |
| Automated Weekly Pipeline | ✅ Complete |
| Email Report Generation | ✅ Complete |

---

## 💡 Key Business Insights

- 📌 Month-to-month customers churn at **42.7%** — 15x higher than two-year contracts
- 📌 Fiber optic users churn at **41.9%** — high price vs perceived value
- 📌 Electronic check users churn at **45.3%** — highest among payment methods
- 📌 New customers (tenure < 12 months) are most vulnerable
- 📌 Estimated yearly revenue lost: **$1,669,570**

---

## ✅ Recommendations

1. Offer annual contract discounts to month-to-month customers
2. Investigate and improve fiber optic service quality
3. Launch proactive onboarding for new customers in first 6 months
4. Promote tech support bundles to reduce churn risk
5. Target 704 high-risk customers with personalized retention campaigns

---

*Teyzix Core Internship — ML-INT-1 | Sara Ahmad | June 2026*
