# %%
import pandas as pd

df = pd.read_csv('E7_data.csv')

print("Descriptive Statistics:")
print(df.describe())

# %%
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# %%
import matplotlib.pyplot as plt

numerical_columns = [col for col in df.columns if col not in ['recorded_day', 'ID']]

# Create a separate scatter plot for each numerical feature
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.scatter(df['id'], df[col], alpha=0.6)
    plt.title(f'Scatter Plot of {col} vs id')
    plt.xlabel('id')
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'scatter_plot_{col}.png')
    plt.show()

# %%
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = df.copy()
df_imputed[df.select_dtypes(include=['float64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64']))

print("\nMissing Values After KNN Imputation:")
print(df_imputed.isnull().sum())

# %%
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

features = [col for col in df_imputed.columns if col not in ['id','recorded_day','raining']]
X = df_imputed[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1])
plt.title('2D t-SNE Plot')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.tight_layout()
plt.savefig('tsne_plot.png')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title('2D PCA Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.savefig('pca_plot.png')
plt.show()

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

features = [col for col in df.columns if col not in ['id','raining']]

Xn = df_imputed[features]
X_scaledn = scaler.fit_transform(Xn)
X_scaled_df = pd.DataFrame(X_scaledn, columns=features)

vif_data = pd.DataFrame()
vif_data['Feature'] = X_scaled_df.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]

print("VIF for each feature:")
print(vif_data)

# %%
columns_to_drop_3 = ['max_temp', 'temperature', 'min_temp']
columns_to_drop_6 = ['max_temp', 'temperature', 'min_temp', 'dew_point', 'air_pressure', 'humidity']
columns_to_drop_8 = ['max_temp', 'temperature', 'min_temp', 'dew_point', 'air_pressure', 'humidity', 'cloud_cover', 'wind_speed']

X_reduced_3 = X_scaled_df.drop(columns=columns_to_drop_3)
X_reduced_6 = X_scaled_df.drop(columns=columns_to_drop_6)
X_reduced_8 = X_scaled_df.drop(columns=columns_to_drop_8)

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

y = df['raining']

def evaluate_model(X, y, model_name='Model'):
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    avg_score = np.mean(scores)
    print(f"{model_name} Average ROC-AUC Score: {avg_score:}")
    return avg_score

roc_auc_full = evaluate_model(X_scaled_df, y, model_name='Full Feature Set')
roc_auc_3 = evaluate_model(X_reduced_3, y, model_name='Reduced (Drop 3 Columns)')
roc_auc_6 = evaluate_model(X_reduced_6, y, model_name='Reduced (Drop 6 Columns)')
roc_auc_8 = evaluate_model(X_reduced_8, y, model_name='Reduced (Drop 8 Columns)')

datasets = {
    'Reduced (Drop 3 Columns)': roc_auc_3,
    'Reduced (Drop 6 Columns)': roc_auc_6,
    'Reduced (Drop 8 Columns)': roc_auc_8
}

labels = list(datasets.keys())
scores = list(datasets.values())

plt.figure(figsize=(8, 6))
plt.scatter(labels, scores, color='blue', label='ROC-AUC Scores')
plt.plot(labels, scores, color='orange', linestyle='--', label='Trend Line')
plt.title('Average ROC-AUC Scores for Logistic Regression Models')
plt.ylabel('Average ROC-AUC Score')
plt.xlabel('Reduced Datasets')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


# %%
from sklearn.decomposition import PCA

pca_1 = PCA(n_components=1)
pca_3 = PCA(n_components=3)
pca_6 = PCA(n_components=6)

X_pca_1 = pca_1.fit_transform(X_scaled_df)
X_pca_3 = pca_3.fit_transform(X_scaled_df)
X_pca_6 = pca_6.fit_transform(X_scaled_df)

# %%
roc_auc_pca_1 = evaluate_model(X_pca_1, y, model_name='PCA (1 Component)')
roc_auc_pca_3 = evaluate_model(X_pca_3, y, model_name='PCA (3 Components)')
roc_auc_pca_6 = evaluate_model(X_pca_6, y, model_name='PCA (6 Components)')

# %%
datasets = {
    'PCA (1 Component)': roc_auc_pca_1,
    'PCA (3 Component)': roc_auc_pca_3,
    'PCA (6 Component)': roc_auc_pca_6
}

pca_configs = list(datasets.keys())
roc_scores_pca = list(datasets.values())
plt.figure(figsize=(8, 6))
plt.scatter(pca_configs, roc_scores_pca, color='blue', label='ROC-AUC Scores')
plt.plot(pca_configs, roc_scores_pca, color='orange', linestyle='--', label='Trend Line')
plt.title('ROC-AUC Scores for PCA Reduced Feature Sets')
plt.xlabel('PCA Configuration')
plt.ylabel('Average ROC-AUC Score')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('roc_auc_pca_scatter_line.png')
plt.show()


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

X_optimal = X_reduced_3
y = df['raining']

lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')

scores_balanced = cross_val_score(lr_balanced, X_optimal, y, cv=5, scoring='roc_auc')
avg_score_balanced = np.mean(scores_balanced)

print("Logistic Regression with class_weight='balanced' Average ROC-AUC Score:", avg_score_balanced)



