"""
Customer Churn Prediction
=========================
A complete ML pipeline for predicting customer churn using:
- Synthetic dataset generation
- Exploratory Data Analysis (EDA)
- Feature engineering & preprocessing
- Multiple model training (Logistic Regression, Random Forest, XGBoost)
- Model evaluation with key metrics
- Churn probability scoring for new customers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate Synthetic Dataset ─────────────────────────────────────────────

def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic telecom-style churn dataset."""
    rng = np.random.default_rng(seed)

    tenure          = rng.integers(1, 72, n_customers)          # months
    monthly_charges = rng.uniform(20, 120, n_customers)
    total_charges   = tenure * monthly_charges * rng.uniform(0.9, 1.1, n_customers)
    num_products    = rng.integers(1, 6, n_customers)
    support_calls   = rng.integers(0, 10, n_customers)
    age             = rng.integers(18, 75, n_customers)

    contract_type   = rng.choice(["Month-to-Month", "One Year", "Two Year"],
                                 n_customers, p=[0.55, 0.25, 0.20])
    payment_method  = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )
    internet_service = rng.choice(["DSL", "Fiber Optic", "No"],
                                  n_customers, p=[0.34, 0.44, 0.22])

    # Churn probability influenced by real-world factors
    churn_score = (
        -0.05 * tenure
        + 0.015 * monthly_charges
        + 0.10 * support_calls
        - 0.30 * num_products
        + (contract_type == "Month-to-Month").astype(float) * 1.5
        + (payment_method == "Electronic Check").astype(float) * 0.5
        + rng.normal(0, 0.5, n_customers)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score + 1))
    churn      = (rng.random(n_customers) < churn_prob).astype(int)

    return pd.DataFrame({
        "CustomerID":      [f"CUST{i:05d}" for i in range(n_customers)],
        "Age":             age,
        "Tenure":          tenure,
        "MonthlyCharges":  monthly_charges.round(2),
        "TotalCharges":    total_charges.round(2),
        "NumProducts":     num_products,
        "SupportCalls":    support_calls,
        "ContractType":    contract_type,
        "PaymentMethod":   payment_method,
        "InternetService": internet_service,
        "Churn":           churn,
    })


# ── 2. Preprocessing ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Encode categoricals, engineer features, split into X / y."""
    data = df.drop(columns=["CustomerID"]).copy()

    # Feature engineering
    data["ChargesPerMonth"]   = data["TotalCharges"] / (data["Tenure"] + 1)
    data["SupportCallsPerMo"] = data["SupportCalls"] / (data["Tenure"] + 1)
    data["HighValueCustomer"] = (data["MonthlyCharges"] > 80).astype(int)

    # Label-encode categorical columns
    cat_cols = ["ContractType", "PaymentMethod", "InternetService"]
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])

    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    return X, y


# ── 3. Model Training ──────────────────────────────────────────────────────────

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_leaf=10, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42
            )),
        ]),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipeline in models.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        pipeline.fit(X_train, y_train)
        y_pred      = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]

        cv_roc      = cross_val_score(pipeline, X_train, y_train,
                                      cv=cv, scoring="roc_auc").mean()
        test_roc    = roc_auc_score(y_test, y_prob)
        avg_prec    = average_precision_score(y_test, y_prob)

        print(classification_report(y_test, y_pred,
                                    target_names=["Retained", "Churned"]))
        print(f"  CV ROC-AUC : {cv_roc:.4f}")
        print(f"  Test ROC-AUC: {test_roc:.4f}")
        print(f"  Avg Precision: {avg_prec:.4f}")

        results[name] = {
            "pipeline":    pipeline,
            "y_pred":      y_pred,
            "y_prob":      y_prob,
            "cv_roc":      cv_roc,
            "test_roc":    test_roc,
            "avg_prec":    avg_prec,
        }

    return results


# ── 4. Visualisations ──────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Churn – Exploratory Data Analysis", fontsize=15, fontweight="bold")

    # Churn distribution
    churn_counts = df["Churn"].value_counts()
    axes[0, 0].pie(churn_counts, labels=["Retained", "Churned"],
                   autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0, 0].set_title("Churn Distribution")

    # Tenure vs Churn
    df.groupby("Churn")["Tenure"].plot(kind="kde", ax=axes[0, 1], legend=True)
    axes[0, 1].set_title("Tenure Distribution by Churn")
    axes[0, 1].set_xlabel("Tenure (months)")
    axes[0, 1].legend(["Retained", "Churned"])

    # Monthly Charges vs Churn
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0, 2])
    axes[0, 2].set_title("Monthly Charges by Churn")
    axes[0, 2].set_xticklabels(["Retained", "Churned"])
    plt.sca(axes[0, 2]); plt.title("Monthly Charges by Churn")

    # Contract type
    contract_churn = df.groupby(["ContractType", "Churn"]).size().unstack(fill_value=0)
    contract_churn.div(contract_churn.sum(axis=1), axis=0).plot(
        kind="bar", ax=axes[1, 0], color=["#4CAF50", "#F44336"], rot=15
    )
    axes[1, 0].set_title("Churn Rate by Contract Type")
    axes[1, 0].set_ylabel("Proportion")
    axes[1, 0].legend(["Retained", "Churned"])

    # Support calls
    support_churn = df.groupby("SupportCalls")["Churn"].mean()
    axes[1, 1].bar(support_churn.index, support_churn.values, color="#2196F3")
    axes[1, 1].set_title("Churn Rate by Support Calls")
    axes[1, 1].set_xlabel("Number of Support Calls")
    axes[1, 1].set_ylabel("Churn Rate")

    # Correlation heatmap
    num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges",
                "NumProducts", "SupportCalls", "Churn"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                ax=axes[1, 2], linewidths=0.5, vmin=-1, vmax=1)
    axes[1, 2].set_title("Correlation Matrix")

    plt.tight_layout()
    plt.savefig("churn_eda.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → churn_eda.png")


def plot_model_comparison(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # ROC curves
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['test_roc']:.3f})", color=color)
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)

    # Precision-Recall curves
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec, prec, label=f"{name} (AP={res['avg_prec']:.3f})", color=color)
    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)

    # AUC bar chart
    names  = list(results.keys())
    aucs   = [r["test_roc"] for r in results.values()]
    bars   = axes[2].bar(names, aucs, color=colors)
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("Test ROC-AUC by Model")
    axes[2].set_ylabel("ROC-AUC")
    for bar, auc in zip(bars, aucs):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f"{auc:.3f}", ha="center")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("churn_models.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → churn_models.png")


def plot_feature_importance(results, feature_names):
    rf_pipeline = results["Random Forest"]["pipeline"]
    rf_model    = rf_pipeline.named_steps["clf"]
    importances = rf_model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2196F3")
    ax.set_title("Top 10 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")
    for bar, val in zip(bars, feat_df["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("churn_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved → churn_features.png")


# ── 5. Inference: Score New Customers ─────────────────────────────────────────

def score_new_customers(best_pipeline, feature_names: list):
    """Score a small batch of new customers."""
    new_customers = pd.DataFrame({
        "Age":             [28, 55, 42],
        "Tenure":          [2, 36, 12],
        "MonthlyCharges":  [95.0, 45.0, 70.0],
        "TotalCharges":    [190.0, 1620.0, 840.0],
        "NumProducts":     [1, 4, 2],
        "SupportCalls":    [7, 1, 3],
        "ContractType":    [0, 2, 1],   # 0=M-t-M, 1=One Year, 2=Two Year
        "PaymentMethod":   [0, 3, 1],   # encoded
        "InternetService": [1, 0, 2],   # encoded
        "ChargesPerMonth":   [95.0,  45.0,  70.0],
        "SupportCallsPerMo": [2.33,  0.028, 0.25],
        "HighValueCustomer": [1,     0,     0],
    })

    probs = best_pipeline.predict_proba(new_customers[feature_names])[:, 1]
    new_customers["ChurnProbability"] = probs.round(4)
    new_customers["RiskLevel"] = pd.cut(
        probs, bins=[0, 0.3, 0.6, 1.0],
        labels=["Low 🟢", "Medium 🟡", "High 🔴"]
    )

    print("\n" + "═" * 55)
    print("  New Customer Churn Scores")
    print("═" * 55)
    print(new_customers[["Tenure", "MonthlyCharges", "SupportCalls",
                          "ChurnProbability", "RiskLevel"]].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   Customer Churn Prediction Pipeline")
    print("=" * 55)

    # 1. Data
    print("\n[1/5] Generating dataset …")
    df = generate_churn_data(5000)
    print(f"      Shape: {df.shape}  |  Churn rate: {df['Churn'].mean():.1%}")

    # 2. EDA
    print("\n[2/5] Plotting EDA …")
    plot_eda(df)

    # 3. Preprocess
    print("\n[3/5] Preprocessing …")
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 4. Train & evaluate
    print("\n[4/5] Training models …")
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 5. Plots
    print("\n[5/5] Plotting results …")
    plot_model_comparison(results, y_test)
    plot_feature_importance(results, X.columns.tolist())

    # Best model
    best_name = max(results, key=lambda k: results[k]["test_roc"])
    print(f"\n  ✅ Best model: {best_name}  (ROC-AUC = {results[best_name]['test_roc']:.4f})")

    # Score new customers
    score_new_customers(results[best_name]["pipeline"], X.columns.tolist())

    print("\n  Done! Plots saved as churn_eda.png, churn_models.png, churn_features.png\n")


if __name__ == "__main__":
    main()
