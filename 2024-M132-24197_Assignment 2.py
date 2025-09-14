# wine_analysis_pipeline.py
# Full pipeline: cleaning, reduction, PCA, t-SNE, visualizations.
# Requirements: numpy, pandas, scikit-learn, matplotlib.
# Run: python wine_analysis_pipeline.py  OR put inside a notebook cell.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans

# --- 0. Load data from the csv file
csv_path = "C:/Users/Admin/Desktop/WORK 2025\MASTERS INFORMATION SYSTEMS WORK/DATA SCIENCE AND VISUALISATION/WineQT.csv"
df = pd.read_csv(csv_path)
print("Loaded dataset shape:", df.shape)
print(df.columns.tolist())

### ---Question 1. Data Cleaning
###----Are there any missing values? How will you handle them?
# Checking for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Identify target (quality)
target_col = "quality" if "quality" in df.columns else None
if target_col:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
else:
    X = df.copy()
    y = None

# If missing values exist, median-impute (safe default)
if X.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy="median")
    X[X.columns] = imputer.fit_transform(X)
    print("Performed median imputation.")

###----Are there outliers or extreme values that need addressing?
# --- 1b. Outlier handling: winsorize (IQR capping)
def winsorize_iqr(df_in):
    df = df_in.copy()
    numeric = df.select_dtypes(include=[np.number])
    lower = numeric.quantile(0.25)
    upper = numeric.quantile(0.75)
    iqr = upper - lower
    lower_bound = lower - 1.5 * iqr
    upper_bound = upper + 1.5 * iqr
    capped = numeric.clip(lower=lower_bound, upper=upper_bound, axis=1)
    df.loc[:, numeric.columns] = capped
    return df

X = winsorize_iqr(X)
print("Winsorized numeric features by IQR.")

###---Do all variables have the same scale? If not, which scaling method will you apply?
# --- 1c. Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Applied StandardScaler.")


###---Question 2. Feature Analysis
###---Which features have low variance and could be removed?
# --- 2. Feature analysis: low variance, correlation-based reduction
# Remove very low variance features
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_scaled)
low_variance = X_scaled.columns[~vt.get_support()].tolist()
print("Low variance features to remove:", low_variance)
X_reduced = pd.DataFrame(vt.transform(X_scaled), columns=X_scaled.columns[vt.get_support()])

###---Are there correlated features that can be reduced?
# Remove one feature from highly correlated pairs (corr > 0.95)
corr_matrix = X_reduced.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
print("Highly correlated features to drop (corr>0.95):", to_drop)
X_reduced = X_reduced.drop(columns=to_drop)
print("Shape after reduction:", X_reduced.shape)

###---Question 3. Dimensionality Reduction
###---Apply at least two different methods (e.g., PCA, UMAP, t-SNE).
# --- Dimensionality reduction using PCA and t-SNE
output_dir = "/mnt/data/embeddings_plots"
os.makedirs(output_dir, exist_ok=True)

embeddings = {}
trust_scores = {}

# PCA 2D
pca2 = PCA(n_components=2, random_state=42)
embeddings['PCA'] = pca2.fit_transform(X_reduced.values)
trust_scores['PCA'] = trustworthiness(X_reduced.values, embeddings['PCA'], n_neighbors=12)

# t-SNE 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
embeddings['t-SNE'] = tsne.fit_transform(X_reduced.values)
trust_scores['t-SNE'] = trustworthiness(X_reduced.values, embeddings['t-SNE'], n_neighbors=12)

###---Compare the results: Which method preserved structure better?
#Comparison of the 2 methods: I computed a standard trustworthiness
#metric (measures how well local neighborhoods are preserved when going
#from original space embedding). The results I produced in the environment
#showed the embeddings and trustworthiness values. Typically and
#in this dataset t-SNE tends to preserve local neighborhood relationships
##better than PCA; PCA preserves global variance structure (principal directions).

###---Question 4. Visualization
###---Create 2D or 3D plots of the reduced dataset.
# --- Visualizations: 2D and 3D embeddings
def plot2d(emb, title, target=None, savepath=None):
    plt.figure(figsize=(8,6))
    if target is not None:
        sc = plt.scatter(emb[:,0], emb[:,1], c=target, s=15)
        plt.colorbar(sc, label=target_col)
    else:
        plt.scatter(emb[:,0], emb[:,1], s=15)
    plt.title(title)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

for name, emb in embeddings.items():
    save_file = os.path.join(output_dir, f"embedding_{name}.png")
    plot2d(emb, f"{name} (2D)", y if y is not None else None, save_file)

# PCA 3D view
pca3 = PCA(n_components=3, random_state=42).fit_transform(X_reduced.values)
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
if y is not None:
    sc = ax.scatter(pca3[:,0], pca3[:,1], pca3[:,2], c=y, s=15)
    fig.colorbar(sc, label=target_col)
else:
    ax.scatter(pca3[:,0], pca3[:,1], pca3[:,2], s=15)
ax.set_title("PCA (3D)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.savefig(os.path.join(output_dir, "embedding_PCA_3D.png"), bbox_inches='tight')
plt.show()

###---Observations from the plots / clustering (what to expect)

#1. PCA (2D) often shows broad separations along principal components — good
# to see global gradients like how features contribute.

#2. t-SNE tends to reveal compact clusters or subgroups (local structure).
# If you color points by quality (target), you may see some separation by
# quality, but often wine quality is overlapping so clusters are not perfectly separated.

#3. PCA 3D allowed a different view of separation — sometimes quality separates
# slightly along PC2 or PC3.

#4. Because we winsorized rather than removed outliers, outliers will have reduced
# influence while still preserving sample counts.

# Quick cluster counts on PCA (example)
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(embeddings['PCA'])
print("Cluster counts on PCA embedding:", np.bincount(clusters))

print("Saved plots to:", output_dir)

###---Question 5. Reflection
#---How did dimensionality reduction affect interpretability?
#---What trade-offs did you face between accuracy and dimensionality?

#Trade-offs / Reflection

#Interpretability vs. dimensionality: PCA is linear and interpretable (PC loadings
# show which original features contribute). t-SNE is non-linear and harder to interpret;
# it excels at finding local clusters but does not provide axes you can easily map back
# to original features.

#Accuracy vs. compression: Reducing to 2D almost always sacrifices some information.
# PCA preserves maximal variance for a given number of components; t-SNE preserves local
# neighborhoods but may distort global distances. Trustworthiness gives a numeric handle
# on how well local neighborhoods were preserved. If you need interpretable components,
# use PCA; if you need to explore cluster structure visually, use t-SNE (or UMAP).

