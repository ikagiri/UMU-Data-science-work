"""
telco_churn_pipeline.py

Full pipeline for Telco Customer Churn dataset:
- First EDA
- Preprocessing
- Feature engineering & reduction
- Second EDA
- Modeling: Decision Tree, Random Forest, XGBoost
- Evaluation: ROC-AUC, accuracy, recall, Stratified K-Fold CV
- Interpretation & suggested retention strategies

Data path (uploaded by user):
/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Outputs saved to /mnt/data/plots and /mnt/data/results.csv
"""

# -----------------------
# Imports & setup
# -----------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve, auc
from sklearn.impute import SimpleImputer

# Ensure reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
PLOTS_DIR = 'C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/Telco Customer Plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
RESULTS_CSV = 'C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/Telco Customer Plots/results_models.csv'

# -----------------------
# 1. Load the dataset
# -----------------------
DATA_PATH = 'C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# 1. FIRST EDA (Raw Data)
# - Summary stats of tenure and MonthlyCharges
# - Churn distribution bar chart
# - Boxplot: churn vs non-churn for MonthlyCharges
# - Correlation heatmap for numerical features
# -----------------------

# Basic info
print(df.info())
print(df.head())

# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Summary stats for tenure and MonthlyCharges
summary = df[['tenure', 'MonthlyCharges']].describe().T
summary['IQR'] = summary['75%'] - summary['25%']
print("Summary stats (tenure and MonthlyCharges):\n", summary)
summary.to_csv(os.path.join(PLOTS_DIR, 'summary_tenure_monthly.csv'))

# Churn distribution bar chart
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn')
plt.title('Churn distribution (0 = No, 1 = Yes)')
plt.savefig(os.path.join(PLOTS_DIR, 'churn_distribution.png'))
plt.close()

# Boxplots: Churn vs MonthlyCharges
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn')
plt.savefig(os.path.join(PLOTS_DIR, 'boxplot_churn_monthlycharges.png'))
plt.close()

# Correlation heatmap for numerical features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# drop the customerID if numeric or other id; ensure numeric features only
if 'customerID' in num_cols:
    num_cols.remove('customerID')
plt.figure(figsize=(10,8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation heatmap (numerical features)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
plt.close()

# -----------------------
# 2. PREPROCESSING
# - Handle missing data
# - Encode categorical variables (One-Hot)
# - Normalize numerical features
# -----------------------

# Quick missing-value check
missing_report = df.isna().sum().sort_values(ascending=False)
print("Missing values per column:\n", missing_report)
missing_report.to_csv(os.path.join(PLOTS_DIR, 'missing_report.csv'))

# In this Telco dataset, 'TotalCharges' sometimes contains spaces -> convert to numeric
if 'TotalCharges' in df.columns:
    # Strip and replace empty strings with NaN, then convert
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Impute numeric missing values with median
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn' in num_features:
    num_features.remove('Churn')  # target excluded
imputer_num = SimpleImputer(strategy='median')
df[num_features] = imputer_num.fit_transform(df[num_features])

# Categorical features handling
cat_features = df.select_dtypes(include=['object']).columns.tolist()
# drop customerID if present as identifier
if 'customerID' in cat_features:
    cat_features.remove('customerID')

print("Numerical features:", num_features)
print("Categorical features:", cat_features)

# One-Hot Encoding for categorical variables (use drop='first' to avoid multicollinearity)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')

# ColumnTransformer to apply scaling to numeric and one-hot to categorical
scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, num_features),
        ('cat', ohe, cat_features)
    ],
    remainder='drop'  # drop other columns like customerID, if any
)

# Build a preprocessed feature matrix X and target y
X_raw = df.drop(columns=['Churn'])
y = df['Churn'].values

# Fit-transform preprocessor to get numeric matrix
X_processed = preprocessor.fit_transform(X_raw)

# Build feature names after OHE
ohe_feature_names = []
if len(cat_features) > 0:
    ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    ohe_feature_names = list(ohe_names)
feature_names = list(num_features) + ohe_feature_names
print("Processed feature matrix shape:", X_processed.shape)
print("Number of processed features:", len(feature_names))

# Save a processed dataframe for later use
X_df = pd.DataFrame(X_processed, columns=feature_names)
X_df.to_csv(os.path.join(PLOTS_DIR, 'X_processed_preview.csv'), index=False)

# -----------------------
# 3. FEATURE ENGINEERING & REDUCTION
# - Create tenure groups (short, medium, long)
# - Total monthly charges per contract type
# - Apply PCA for numerical features
# -----------------------

# Tenure groups (custom thresholds)
def tenure_group(tenure):
    if tenure <= 12:
        return 'short'
    elif tenure <= 48:
        return 'medium'
    else:
        return 'long'

# Attach tenure group to original df for aggregation and also to encoded matrix
df['tenure_group'] = df['tenure'].apply(tenure_group)
# Save counts
df['tenure_group'].value_counts().to_csv(os.path.join(PLOTS_DIR, 'tenure_group_counts.csv'))

# Total monthly charges per contract type
# We assume MonthlyCharges is per customer; 'Total monthly charges per contract type' = sum MonthlyCharges grouped by Contract
if 'Contract' in df.columns:
    total_monthly_by_contract = df.groupby('Contract')['MonthlyCharges'].sum().reset_index().rename(columns={'MonthlyCharges':'TotalMonthlyCharges'})
    total_monthly_by_contract.to_csv(os.path.join(PLOTS_DIR, 'total_monthly_by_contract.csv'), index=False)

    # bar plot
    plt.figure(figsize=(6,4))
    sns.barplot(data=total_monthly_by_contract, x='Contract', y='TotalMonthlyCharges')
    plt.title('Total Monthly Charges by Contract Type')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'total_monthly_by_contract.png'))
    plt.close()

# PCA on numerical features only (for variance capture)
# We'll use the scaled numeric columns from the preprocessor (first len(num_features) columns)
X_num_scaled = X_processed[:, :len(num_features)]
pca = PCA(n_components=min(10, X_num_scaled.shape[1]), random_state=RANDOM_STATE)
pca_components = pca.fit_transform(X_num_scaled)
explained_variance = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained_variance)

# Save PCA explained variance
pca_df = pd.DataFrame({
    'PC': np.arange(1, len(explained_variance)+1),
    'explained_variance': explained_variance,
    'cumulative_explained': cum_explained
})
pca_df.to_csv(os.path.join(PLOTS_DIR, 'pca_explained_variance.csv'), index=False)

# Plot PCA explained variance
plt.figure(figsize=(6,4))
plt.plot(pca_df['PC'], pca_df['cumulative_explained'], marker='o')
plt.xlabel('Number of PCA components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA cumulative explained variance (numerical features)')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'pca_cumulative_variance.png'))
plt.close()

# -----------------------
# 4. SECOND EDA (Post-Processing)
# - Feature importance visualization (Random Forest)
# - PCA explained variance chart (saved above)
# -----------------------

# Train a Random Forest quickly to obtain feature importances (we will later do full modeling)
rf_quick = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf_quick.fit(X_processed, y)

# Extract importances and map to feature names
rf_importances = pd.Series(rf_quick.feature_importances_, index=feature_names).sort_values(ascending=False)
rf_importances.head(20).to_csv(os.path.join(PLOTS_DIR, 'rf_top20_importances.csv'))

# Plot top 20 importances
plt.figure(figsize=(8,10))
sns.barplot(x=rf_importances.head(20).values, y=rf_importances.head(20).index)
plt.title('Top 20 feature importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_top20_importances.png'))
plt.close()

# -----------------------
# 5. ML MODELING
# - Train Decision Tree, Random Forest, XGBoost
# - We'll do train/test split and also Stratified K-Fold CV in evaluation step
# -----------------------
X = X_processed
y = df['Churn'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=RANDOM_STATE)

# Instantiate models
dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
rf_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)

# Fit models
print("Training Decision Tree...")
dt_model.fit(X_train, y_train)
print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# -----------------------
# 6. EVALUATION
# - Compute ROC-AUC, accuracy, recall on test set
# - Do Stratified K-Fold cross-validation and report mean/std for ROC-AUC and accuracy
# -----------------------

def evaluate_on_test(model, X_test, y_test, model_name):
    # Predict probabilities and classes
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        # For models without predict_proba fallback to decision_function or binary predict
        try:
            y_proba = model.decision_function(X_test)
            # scale to [0,1] using minmax if necessary
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)
        except Exception:
            y_proba = model.predict(X_test)
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"{model_name} -- ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}, Recall: {rec:.4f}")
    return {'model': model_name, 'roc_auc': roc_auc, 'accuracy': acc, 'recall': rec, 'y_proba': y_proba, 'y_pred': y_pred}

res_dt = evaluate_on_test(dt_model, X_test, y_test, "DecisionTree")
res_rf = evaluate_on_test(rf_model, X_test, y_test, "RandomForest")
res_xgb = evaluate_on_test(xgb_model, X_test, y_test, "XGBoost")

# Save ROC curves
plt.figure(figsize=(6,5))
for res, color in zip([res_dt, res_rf, res_xgb], ['orange','green','blue']):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label=f"{res['model']} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Test Set')
plt.legend(loc='lower right')
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves_models.png'))
plt.close()

# Stratified K-Fold CV for ROC-AUC and Accuracy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models_for_cv = {
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
}

cv_results = []
for name, model in models_for_cv.items():
    print(f"Running Stratified CV for {name} ...")
    # Use cross_val_score for accuracy and ROC-AUC (roc_auc requires prob estimates)
    # For roc_auc scoring with cross_val_score, sklearn will call predict_proba if available.
    try:
        roc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    except Exception as e:
        print(f"CV scoring failed for {name}: {e}")
        roc_scores = np.array([np.nan])
        acc_scores = np.array([np.nan])
    cv_results.append({
        'model': name,
        'roc_auc_mean': np.nanmean(roc_scores),
        'roc_auc_std': np.nanstd(roc_scores),
        'accuracy_mean': np.nanmean(acc_scores),
        'accuracy_std': np.nanstd(acc_scores)
    })
    print(f"{name} CV ROC-AUC mean±std: {np.nanmean(roc_scores):.4f} ± {np.nanstd(roc_scores):.4f}")
    print(f"{name} CV Accuracy mean±std: {np.nanmean(acc_scores):.4f} ± {np.nanstd(acc_scores):.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(os.path.join(PLOTS_DIR, 'cv_results_summary.csv'), index=False)

# Save test evaluation results
test_eval_df = pd.DataFrame([
    {'model': res_dt['model'], 'roc_auc': res_dt['roc_auc'], 'accuracy': res_dt['accuracy'], 'recall': res_dt['recall']},
    {'model': res_rf['model'], 'roc_auc': res_rf['roc_auc'], 'accuracy': res_rf['accuracy'], 'recall': res_rf['recall']},
    {'model': res_xgb['model'], 'roc_auc': res_xgb['roc_auc'], 'accuracy': res_xgb['accuracy'], 'recall': res_xgb['recall']}
])
test_eval_df.to_csv(RESULTS_CSV, index=False)
print("Saved test evaluation results to", RESULTS_CSV)

# -----------------------
# 7. INTERPRETATION
# - Identify top churn drivers (from Random Forest importances)
# - Suggest retention strategies
# -----------------------

# Use rf_quick importances (or rf_model importances)
rf_importances_full = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
top_drivers = rf_importances_full.head(15)
top_drivers.to_csv(os.path.join(PLOTS_DIR, 'top_churn_drivers.csv'))

# Print top drivers
print("Top churn drivers (by Random Forest importance):")
for feat, val in top_drivers.items():
    print(f"{feat}: {val:.4f}")

# Save textual suggestions for retention strategies
suggestions = """
Retention strategy suggestions (based on common telco churn drivers and feature importances):

1. Target high-risk tenure groups:
   - Short-tenure customers (<=12 months) often churn more. Offer onboarding incentives
     (discounted rates, free trials of valued features, loyalty program enrollment).

2. Pricing & Contract incentives:
   - If Month-to-month customers show higher churn, promote annual/2-year contracts with
     small discounts, or offer flexible mid-term discounts to reduce churn.

3. Improve product/feature satisfaction:
   - If features like 'InternetService', 'TechSupport', or 'OnlineSecurity' are important,
     improve service quality and upsell valued features with bundled savings.

4. Focus on high MonthlyCharges or increases:
   - Customers with unexpectedly high bills may churn. Proactive billing transparency,
     personalized plan recommendations, and billing alerts can reduce surprises.

5. Customer support & retention offers:
   - Proactively reach out to at-risk customers with tailored retention offers,
     priority support, or temporary discounts.

6. Monitor payment issues:
   - Offer alternate billing options or grace periods to customers with payment issues.

Note: Always A/B test any retention offers to measure ROI and avoid over-subsidizing customers.
"""

with open(os.path.join(PLOTS_DIR, 'retention_suggestions.txt'), 'w') as f:
    f.write(suggestions)

print("Saved retention suggestions at", os.path.join(PLOTS_DIR, 'retention_suggestions.txt'))

# Also save top RF importances plot (already saved earlier as rf_top20_importances.png but create a plot for final rf_model)
plt.figure(figsize=(8,10))
rf_importances_full.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 20 feature importances (final Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_top20_importances_final.png'))
plt.close()

# -----------------------
# End: summary print
# -----------------------
print("\nPipeline complete.")
print("Plots and CSV outputs saved to:", PLOTS_DIR)
print("Model test results saved to:", RESULTS_CSV)
