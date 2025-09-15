# breast_cancer_analysis.py
# Full reproducible pipeline: EDA, preprocessing, feature engineering, reduction, models, evaluation
# Run with: python breast_cancer_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Uncomment the next import if you have umap installed: pip install umap-learn
import umap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve)
from sklearn.calibration import calibration_curve

# ---------- Config ----------
CSV_PATH = "C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/Breast Cancer Winscon.csv"
PLOTS_DIR = "C:/Users/Admin/Desktop/WORK 2025/MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION"
os.makedirs(PLOTS_DIR, exist_ok=True)
# ----------------------------

# Load data
df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns.tolist())
# In this dataset 'Class' holds 2 (benign) and 4 (malignant)
TARGET = "Class"

# Map target: 2 -> 0 (benign), 4 -> 1 (malignant)
df[TARGET] = df[TARGET].replace({2: 0, 4: 1}).astype(int)
y = df[TARGET].values

# Drop ID column if present
if 'Sample_code_number' in df.columns:
    df = df.drop(columns=['Sample_code_number'])

# Prepare features (coerce to numeric where needed)
X = df.drop(columns=[TARGET]).copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors='coerce')

# 1) First EDA (Raw Data)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
summary = pd.DataFrame(index=num_cols)
summary['mean'] = X[num_cols].mean()
summary['std'] = X[num_cols].std()
summary['25%'] = X[num_cols].quantile(0.25)
summary['50%'] = X[num_cols].median()
summary['75%'] = X[num_cols].quantile(0.75)
summary['IQR'] = summary['75%'] - summary['25%']
print("Summary stats (first rows):\n", summary.head())

missing = X.isna().sum().to_frame("missing_count")
missing['missing_pct'] = missing['missing_count'] / len(X) * 100
print("Missing values:\n", missing[missing['missing_count'] > 0])

# Histograms and boxplots
for col in num_cols:
    plt.figure(figsize=(6,4))
    plt.hist(X[col].dropna(), bins=30)
    plt.title(f"Histogram - {col}")
    plt.xlabel(col); plt.ylabel("Count")
    plt.savefig(os.path.join(PLOTS_DIR, f"hist_{col}.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    data = pd.concat([X[col], pd.Series(y, name='target')], axis=1)
    data.boxplot(column=col, by='target')
    plt.title(f"Boxplot {col} by target")
    plt.suptitle('')
    plt.savefig(os.path.join(PLOTS_DIR, f"box_{col}.png"))
    plt.close()

# Scatter of top 2-3 by variance
variances = X[num_cols].var().sort_values(ascending=False)
top_feats = variances.index[:3].tolist()
pairs = []
if len(top_feats) >= 2:
    if len(top_feats) >= 3:
        pairs = [(top_feats[0], top_feats[1]), (top_feats[0], top_feats[2]), (top_feats[1], top_feats[2])]
    else:
        pairs = [(top_feats[0], top_feats[1])]
for a,b in pairs:
    plt.figure(figsize=(6,4))
    plt.scatter(X[a], X[b], c=y)
    plt.title(f"Scatter {a} vs {b} (target colored)")
    plt.savefig(os.path.join(PLOTS_DIR, f"scatter_{a}_vs_{b}.png"))
    plt.close()

# 2) Preprocessing
# Impute missing with median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X[num_cols]), columns=num_cols)

# Scale continuous features (Standard scaler recommended for many models)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=num_cols)

# 3) Feature Engineering & Reduction
# Example: generate simple ratios (avoid divide by zero)
X_feat = X_scaled.copy()
for i in range(min(3, len(num_cols)-1)):
    a = num_cols[i]; b = num_cols[i+1]
    X_feat[f"ratio_{a}_to_{b}"] = (X_imputed[a] / (X_imputed[b].replace(0, np.nan))).fillna(0)

# RFE with RandomForest to pick top features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
n_select = min(10, X_feat.shape[1])
rfe = RFE(estimator=rf, n_features_to_select=n_select, step=1)
rfe.fit(X_feat, y)
selected = X_feat.columns[rfe.support_].tolist()
print("RFE selected features:", selected)

# PCA
pca = PCA(n_components=min(10, X_feat.shape[1]))
pca_res = pca.fit_transform(X_feat)
explained = pca.explained_variance_ratio_
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(explained))
plt.xlabel("Number of PCA components"); plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance")
plt.savefig(os.path.join(PLOTS_DIR, "pca_cumulative_variance.png")); plt.close()

# t-SNE (2D visualization)
# NOTE: t-SNE can be slow on large datasets; reduce dimensions with PCA first for speed if needed.
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
tsne2 = tsne.fit_transform(X_feat)
plt.figure(figsize=(6,4))
plt.scatter(tsne2[:,0], tsne2[:,1], c=y)
plt.title("t-SNE 2D projection (color=target)")
plt.savefig(os.path.join(PLOTS_DIR, "tsne_2d.png")); plt.close()

# Optional: UMAP (fast) - requires 'umap-learn' package (pip install umap-learn)
try:
    import umap
    umap2 = umap.UMAP(n_components=2, random_state=42).fit_transform(X_feat)
    plt.figure(figsize=(6,4))
    plt.scatter(umap2[:,0], umap2[:,1], c=y)
    plt.title("UMAP 2D projection (color=target)")
    plt.savefig(os.path.join(PLOTS_DIR, "umap_2d.png")); plt.close()
except Exception as e:
    print("UMAP not run (package missing or error):", e)

# 4) Second EDA (Post-Processing)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_feat, y)
importances = pd.Series(dt.feature_importances_, index=X_feat.columns).sort_values(ascending=False)
print("Top feature importances (Decision Tree):\n", importances.head(15))
# Save top importances plot
plt.figure(figsize=(10,6))
importances.head(20).plot(kind='bar')
plt.title("Decision Tree top feature importances")
plt.savefig(os.path.join(PLOTS_DIR, "feature_importances.png")); plt.close()

# 5) ML Modeling: Naive Bayes and k-NN
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.25, random_state=42, stratify=y)
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
gnb.fit(X_train, y_train)
knn.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)
y_proba_gnb = gnb.predict_proba(X_test)[:,1]

y_pred_knn = knn.predict(X_test)
y_proba_knn = knn.predict_proba(X_test)[:,1] if hasattr(knn, "predict_proba") else None

# 6) Evaluation
def evaluate(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp+fn)>0 else np.nan
    specificity = tn / (tn + fp) if (tn+fp)>0 else np.nan
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return dict(accuracy=acc, sensitivity=sensitivity, specificity=specificity, roc_auc=roc_auc, tn=tn, fp=fp, fn=fn, tp=tp)

res_gnb = evaluate(y_test, y_pred_gnb, y_proba_gnb)
res_knn = evaluate(y_test, y_pred_knn, y_proba_knn)
print("Model evaluation:\n", pd.DataFrame([res_gnb, res_knn], index=['GNB','kNN']))

# ROC curves
plt.figure(figsize=(6,4))
fpr, tpr, _ = roc_curve(y_test, y_proba_gnb)
plt.plot(fpr, tpr, label=f'GNB AUC={res_gnb["roc_auc"]:.3f}')
if y_proba_knn is not None:
    fpr2, tpr2, _ = roc_curve(y_test, y_proba_knn)
    plt.plot(fpr2, tpr2, label=f'kNN AUC={res_knn["roc_auc"]:.3f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png")); plt.close()

# Calibration curves
plt.figure(figsize=(6,4))
p_true_g, p_pred_g = calibration_curve(y_test, y_proba_gnb, n_bins=10)
plt.plot(p_pred_g, p_true_g, marker='o', label='GNB')
if y_proba_knn is not None:
    p_true_k, p_pred_k = calibration_curve(y_test, y_proba_knn, n_bins=10)
    plt.plot(p_pred_k, p_true_k, marker='o', label='kNN')
plt.plot([0,1],[0,1],'--')
plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
plt.title("Calibration curves"); plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_curves.png")); plt.close()

print("Plots and results saved under:", PLOTS_DIR)
