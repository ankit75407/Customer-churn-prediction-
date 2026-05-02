# 📉 Customer Churn Prediction

A complete end-to-end machine learning pipeline for predicting customer churn using Python and scikit-learn. The project covers data generation, exploratory data analysis, feature engineering, model training, evaluation, and real-time inference scoring.

---

## 📁 Project Structure

```
churn-prediction/
│
├── churn_prediction.py       # Main pipeline script
├── churn_dataset.csv         # Generated dataset (5,000 customers)
├── churn_eda.png             # EDA visualizations
├── churn_models.png          # Model comparison plots
├── churn_features.png        # Feature importance chart
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.8+** installed.

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Pipeline

```bash
python churn_prediction.py
```

---

## 📊 Dataset

The dataset is synthetically generated to simulate a telecom company's customer base.

| Column           | Type    | Description                                      |
|------------------|---------|--------------------------------------------------|
| CustomerID       | String  | Unique customer identifier                       |
| Age              | Integer | Customer age (18–74)                             |
| Tenure           | Integer | Months as a customer (1–71)                      |
| MonthlyCharges   | Float   | Monthly bill amount ($20–$120)                   |
| TotalCharges     | Float   | Cumulative total charges                         |
| NumProducts      | Integer | Number of products subscribed (1–5)              |
| SupportCalls     | Integer | Number of support calls made (0–9)               |
| ContractType     | String  | Month-to-Month / One Year / Two Year             |
| PaymentMethod    | String  | Electronic Check / Mailed Check / Bank Transfer / Credit Card |
| InternetService  | String  | DSL / Fiber Optic / No                           |
| Churn            | Integer | Target variable — 1 = Churned, 0 = Retained      |

- **Total Records:** 5,000
- **Churn Rate:** ~30.4%
- **Class Split:** 70% Retained / 30% Churned

---

## 🔧 Pipeline Stages

### 1. Data Generation
Synthetic data is created using NumPy with realistic churn drivers. Churn probability is modeled using a logistic function influenced by tenure, charges, support calls, contract type, and payment method.

### 2. Exploratory Data Analysis (EDA)
Six visualizations are generated:
- Churn distribution (pie chart)
- Tenure distribution by churn status (KDE)
- Monthly charges by churn (box plot)
- Churn rate by contract type (bar chart)
- Churn rate by support call count (bar chart)
- Feature correlation heatmap

### 3. Feature Engineering
Three new features are derived from existing data:

| Feature             | Formula                                  |
|---------------------|------------------------------------------|
| ChargesPerMonth     | TotalCharges / (Tenure + 1)              |
| SupportCallsPerMo   | SupportCalls / (Tenure + 1)              |
| HighValueCustomer   | 1 if MonthlyCharges > $80, else 0        |

### 4. Preprocessing
- Categorical variables are label-encoded
- Data is split 80% train / 20% test with stratification
- Pipelines handle imputation and scaling internally

### 5. Model Training

| Model                | Key Parameters                                      |
|----------------------|-----------------------------------------------------|
| Logistic Regression  | C=1.0, max_iter=1000                                |
| Random Forest        | 200 trees, max_depth=8, min_samples_leaf=10         |
| Gradient Boosting    | 200 estimators, learning_rate=0.05, max_depth=4     |

### 6. Evaluation Metrics
- ROC-AUC (test + 5-fold cross-validation)
- Average Precision Score
- Classification Report (precision, recall, F1)
- Confusion Matrix

### 7. Inference
The best model scores new customers and outputs:
- **Churn Probability** (0.0 – 1.0)
- **Risk Level:** Low 🟢 / Medium 🟡 / High 🔴

---

## 📈 Output Plots

| File                 | Description                                          |
|----------------------|------------------------------------------------------|
| `churn_eda.png`      | 6-panel EDA visualization                            |
| `churn_models.png`   | ROC curves, precision-recall curves, AUC bar chart   |
| `churn_features.png` | Top 10 feature importances from Random Forest        |

---

## 🧠 Model Performance (Typical Results)

| Model                | CV ROC-AUC | Test ROC-AUC |
|----------------------|------------|--------------|
| Logistic Regression  | ~0.79      | ~0.80        |
| Random Forest        | ~0.84      | ~0.85        |
| Gradient Boosting    | ~0.85      | ~0.86        |

> Results may vary slightly due to random seed effects.

---

## 🔄 Using Your Own Data

To use a real dataset instead of the synthetic one, replace the data generation step:

```python
# Replace this:
df = generate_churn_data(5000)

# With this:
df = pd.read_csv("your_data.csv")
```

Ensure your CSV has a `Churn` column (0 or 1) and update column names in `preprocess()` to match your data.

---

## 📦 Dependencies

| Library      | Version  | Purpose                        |
|--------------|----------|--------------------------------|
| numpy        | ≥ 1.23   | Numerical computation          |
| pandas       | ≥ 1.5    | Data manipulation              |
| matplotlib   | ≥ 3.6    | Plotting                       |
| seaborn      | ≥ 0.12   | Statistical visualizations     |
| scikit-learn | ≥ 1.2    | ML models & preprocessing      |

---

## 📝 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙌 Acknowledgements

Inspired by real-world telecom churn datasets such as the IBM Watson Telco Customer Churn dataset. Built entirely with open-source Python libraries.
