# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T08:57:56.405633Z","iopub.execute_input":"2025-01-16T08:57:56.406043Z","iopub.status.idle":"2025-01-16T08:58:00.924593Z","shell.execute_reply.started":"2025-01-16T08:57:56.406011Z","shell.execute_reply":"2025-01-16T08:58:00.923555Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import optuna
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
import os
from datetime import datetime

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T08:59:44.542491Z","iopub.execute_input":"2025-01-16T08:59:44.542912Z","iopub.status.idle":"2025-01-16T08:59:44.691392Z","shell.execute_reply.started":"2025-01-16T08:59:44.542882Z","shell.execute_reply":"2025-01-16T08:59:44.690292Z"}}
# Load the dataset
file_path = "kacper_reg_739582(1).csv"  # Update this with your file path
data = pd.read_csv(file_path)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T08:59:45.980595Z","iopub.execute_input":"2025-01-16T08:59:45.980986Z","iopub.status.idle":"2025-01-16T08:59:45.986922Z","shell.execute_reply.started":"2025-01-16T08:59:45.980954Z","shell.execute_reply":"2025-01-16T08:59:45.985688Z"}}
data.shape

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T08:59:47.168461Z","iopub.execute_input":"2025-01-16T08:59:47.168885Z","iopub.status.idle":"2025-01-16T08:59:47.175275Z","shell.execute_reply.started":"2025-01-16T08:59:47.168817Z","shell.execute_reply":"2025-01-16T08:59:47.174244Z"}}
data.columns

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T08:59:48.311648Z","iopub.execute_input":"2025-01-16T08:59:48.312033Z","iopub.status.idle":"2025-01-16T08:59:48.331225Z","shell.execute_reply.started":"2025-01-16T08:59:48.312002Z","shell.execute_reply":"2025-01-16T08:59:48.329930Z"}}
# Encode categorical target variable 'uch'
label_encoder = LabelEncoder()
data['uch_encoded'] = label_encoder.fit_transform(data['uch'])

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:04.494293Z","iopub.execute_input":"2025-01-16T09:00:04.494660Z","iopub.status.idle":"2025-01-16T09:00:04.520986Z","shell.execute_reply.started":"2025-01-16T09:00:04.494631Z","shell.execute_reply":"2025-01-16T09:00:04.519941Z"}}
data = data.drop(columns=['cell_count'])
data = data.drop(columns=['uid'], errors='ignore')
data = data.drop(columns=['chem_id'], errors='ignore')
data = data.drop(columns=['animal'], errors='ignore')
data = data.drop(columns=['novelty'], errors='ignore')

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:05.481069Z","iopub.execute_input":"2025-01-16T09:00:05.481399Z","iopub.status.idle":"2025-01-16T09:00:05.487780Z","shell.execute_reply.started":"2025-01-16T09:00:05.481374Z","shell.execute_reply":"2025-01-16T09:00:05.486806Z"}}
data.columns

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:12.952176Z","iopub.execute_input":"2025-01-16T09:00:12.952513Z","iopub.status.idle":"2025-01-16T09:00:12.959247Z","shell.execute_reply.started":"2025-01-16T09:00:12.952486Z","shell.execute_reply":"2025-01-16T09:00:12.958308Z"}}
# Split data into features (X) and target (y)
X = data.drop(columns=['uch', 'uch_encoded'], errors='ignore')
y = data['uch_encoded']

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:14.019169Z","iopub.execute_input":"2025-01-16T09:00:14.019522Z","iopub.status.idle":"2025-01-16T09:00:14.025451Z","shell.execute_reply.started":"2025-01-16T09:00:14.019491Z","shell.execute_reply":"2025-01-16T09:00:14.024509Z"}}
X.columns

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:16.428220Z","iopub.execute_input":"2025-01-16T09:00:16.428548Z","iopub.status.idle":"2025-01-16T09:00:16.435418Z","shell.execute_reply.started":"2025-01-16T09:00:16.428524Z","shell.execute_reply":"2025-01-16T09:00:16.434546Z"}}
y

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:18.360123Z","iopub.execute_input":"2025-01-16T09:00:18.360475Z","iopub.status.idle":"2025-01-16T09:00:18.373162Z","shell.execute_reply.started":"2025-01-16T09:00:18.360445Z","shell.execute_reply":"2025-01-16T09:00:18.371748Z"}}
# Convert categorical columns in X to numerical
X = pd.get_dummies(X, drop_first=True)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:47.035112Z","iopub.execute_input":"2025-01-16T09:00:47.035459Z","iopub.status.idle":"2025-01-16T09:00:47.081231Z","shell.execute_reply.started":"2025-01-16T09:00:47.035429Z","shell.execute_reply":"2025-01-16T09:00:47.080138Z"}}
# Step 1: Preprocessing
# Standardize numerical features
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:48.863096Z","iopub.execute_input":"2025-01-16T09:00:48.863433Z","iopub.status.idle":"2025-01-16T09:00:48.869936Z","shell.execute_reply.started":"2025-01-16T09:00:48.863407Z","shell.execute_reply":"2025-01-16T09:00:48.868813Z"}}
# Check for non-numeric columns in X
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

if not non_numeric_cols.empty:
    print(f"Non-numeric columns found: {non_numeric_cols}")
    for col in non_numeric_cols:
        print(f"Unique values in '{col}': {X[col].unique()}")

    raise ValueError("Some features are still non-numeric after encoding.")

# Verify all columns are numeric
if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    raise ValueError("Some features are still non-numeric after encoding.")

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:50.629546Z","iopub.execute_input":"2025-01-16T09:00:50.629912Z","iopub.status.idle":"2025-01-16T09:00:50.649181Z","shell.execute_reply.started":"2025-01-16T09:00:50.629884Z","shell.execute_reply":"2025-01-16T09:00:50.648136Z"}}
# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:53.271647Z","iopub.execute_input":"2025-01-16T09:00:53.272047Z","iopub.status.idle":"2025-01-16T09:00:53.277572Z","shell.execute_reply.started":"2025-01-16T09:00:53.272016Z","shell.execute_reply":"2025-01-16T09:00:53.276436Z"}}
# Optuna optimization for Random Forest

def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:53.783470Z","iopub.execute_input":"2025-01-16T09:00:53.783809Z","iopub.status.idle":"2025-01-16T09:00:53.788802Z","shell.execute_reply.started":"2025-01-16T09:00:53.783784Z","shell.execute_reply":"2025-01-16T09:00:53.787803Z"}}
# Optuna optimization for Logistic Regression

def objective_lr(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:55.429210Z","iopub.execute_input":"2025-01-16T09:00:55.429584Z","iopub.status.idle":"2025-01-16T09:00:55.435090Z","shell.execute_reply.started":"2025-01-16T09:00:55.429552Z","shell.execute_reply":"2025-01-16T09:00:55.433970Z"}}
# Optuna optimization for Support Vector Machine

def objective_svc(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:55.975747Z","iopub.execute_input":"2025-01-16T09:00:55.976168Z","iopub.status.idle":"2025-01-16T09:00:55.981134Z","shell.execute_reply.started":"2025-01-16T09:00:55.976137Z","shell.execute_reply":"2025-01-16T09:00:55.980119Z"}}
# Optuna optimization for k-Nearest Neighbors
def objective_knn(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:57.218683Z","iopub.execute_input":"2025-01-16T09:00:57.219081Z","iopub.status.idle":"2025-01-16T09:00:57.223805Z","shell.execute_reply.started":"2025-01-16T09:00:57.219051Z","shell.execute_reply":"2025-01-16T09:00:57.222690Z"}}
# Optuna optimization for Naive Bayes
def objective_nb(trial):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:57.463468Z","iopub.execute_input":"2025-01-16T09:00:57.463895Z","iopub.status.idle":"2025-01-16T09:00:57.469106Z","shell.execute_reply.started":"2025-01-16T09:00:57.463855Z","shell.execute_reply":"2025-01-16T09:00:57.467887Z"}}
# Optuna optimization for Decision Tree
def objective_dt(trial):
    max_depth = trial.suggest_int("max_depth", 2, 20)
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:58.472240Z","iopub.execute_input":"2025-01-16T09:00:58.472617Z","iopub.status.idle":"2025-01-16T09:00:58.477689Z","shell.execute_reply.started":"2025-01-16T09:00:58.472585Z","shell.execute_reply":"2025-01-16T09:00:58.476427Z"}}
# Optuna optimization for Gradient Boosting
def objective_gb(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:00:59.398967Z","iopub.execute_input":"2025-01-16T09:00:59.399339Z","iopub.status.idle":"2025-01-16T09:00:59.405669Z","shell.execute_reply.started":"2025-01-16T09:00:59.399309Z","shell.execute_reply":"2025-01-16T09:00:59.404545Z"}}
# Optuna optimization for LightGBM
def objective_lgb(trial):
    num_classes = len(np.unique(y_train))  # Calculate the number of unique classes
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "num_class": num_classes  # Specify the number of classes
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:01:01.131208Z","iopub.execute_input":"2025-01-16T09:01:01.131554Z","iopub.status.idle":"2025-01-16T09:01:01.137914Z","shell.execute_reply.started":"2025-01-16T09:01:01.131528Z","shell.execute_reply":"2025-01-16T09:01:01.136372Z"}}
# Optuna optimization for XGBoost
def objective_xgb(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
    }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:01:02.019540Z","iopub.execute_input":"2025-01-16T09:01:02.019924Z","iopub.status.idle":"2025-01-16T09:01:02.025434Z","shell.execute_reply.started":"2025-01-16T09:01:02.019894Z","shell.execute_reply":"2025-01-16T09:01:02.024288Z"}}
# Optuna optimization for CatBoost
def objective_cb(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "iterations": trial.suggest_int("iterations", 10, 200),
    }
    
    model = CatBoostClassifier(**params, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:01:02.942809Z","iopub.execute_input":"2025-01-16T09:01:02.943184Z","iopub.status.idle":"2025-01-16T09:01:02.949403Z","shell.execute_reply.started":"2025-01-16T09:01:02.943157Z","shell.execute_reply":"2025-01-16T09:01:02.947931Z"}}
# Optuna optimization for ElasticNet
def objective_en(trial):
    alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    y_pred = np.round(model.predict(X_test))  # Round predictions for classification
    return accuracy_score(y_test, y_pred)

# %% [code]
# Run Optuna for each model
print("Optimizing Random Forest...")
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)
print("Best parameters for Random Forest:", study_rf.best_params)
print("Best accuracy for Random Forest:", study_rf.best_value)

# %% [code]
print("Optimizing Logistic Regression...")
study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_lr, n_trials=50)
print("Best parameters for Logistic Regression:", study_lr.best_params)
print("Best accuracy for Logistic Regression:", study_lr.best_value)

# %% [code]
print("Optimizing k-Nearest Neighbors...")
study_knn = optuna.create_study(direction="maximize")
study_knn.optimize(objective_knn, n_trials=50)
print("Best parameters for k-Nearest Neighbors:", study_knn.best_params)
print("Best accuracy for k-Nearest Neighbors:", study_knn.best_value)

# %% [code]
print("Optimizing Naive Bayes...")
study_nb = optuna.create_study(direction="maximize")
study_nb.optimize(objective_nb, n_trials=50)
print("Best accuracy for Naive Bayes:", study_nb.best_value)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:01:08.623679Z","iopub.execute_input":"2025-01-16T09:01:08.624101Z","iopub.status.idle":"2025-01-16T09:01:21.536840Z","shell.execute_reply.started":"2025-01-16T09:01:08.624071Z","shell.execute_reply":"2025-01-16T09:01:21.535953Z"}}
print("Optimizing Decision Tree...")
study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective_dt, n_trials=50)
print("Best parameters for Decision Tree:", study_dt.best_params)
print("Best accuracy for Decision Tree:", study_dt.best_value)

# %% [code]
print("Optimizing Gradient Boosting...")
study_gb = optuna.create_study(direction="maximize")
study_gb.optimize(objective_gb, n_trials=50)
print("Best parameters for Gradient Boosting:", study_gb.best_params)
print("Best accuracy for Gradient Boosting:", study_gb.best_value)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:01:26.511434Z","iopub.execute_input":"2025-01-16T09:01:26.511776Z","iopub.status.idle":"2025-01-16T09:31:23.469323Z","shell.execute_reply.started":"2025-01-16T09:01:26.511748Z","shell.execute_reply":"2025-01-16T09:31:23.468168Z"}}
print("Optimizing LightGBM...")
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=50)
print("Best parameters for LightGBM:", study_lgb.best_params)
print("Best accuracy for LightGBM:", study_lgb.best_value)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-16T09:31:23.470948Z","iopub.execute_input":"2025-01-16T09:31:23.471419Z","iopub.status.idle":"2025-01-16T09:49:12.236046Z","shell.execute_reply.started":"2025-01-16T09:31:23.471379Z","shell.execute_reply":"2025-01-16T09:49:12.234717Z"}}
print("Optimizing XGBoost...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=50)
print("Best parameters for XGBoost:", study_xgb.best_params)
print("Best accuracy for XGBoost:", study_xgb.best_value)

# %% [code]
print("Optimizing CatBoost...")
study_cb = optuna.create_study(direction="maximize")
study_cb.optimize(objective_cb, n_trials=50)
print("Best parameters for CatBoost:", study_cb.best_params)
print("Best accuracy for CatBoost:", study_cb.best_value)

# %% [code]
print("Optimizing ElasticNet...")
study_en = optuna.create_study(direction="maximize")
study_en.optimize(objective_en, n_trials=50)
print("Best parameters for ElasticNet:", study_en.best_params)
print("Best accuracy for ElasticNet:", study_en.best_value)

# %% [code]
# After the data preprocessing and before model training, add PCA analysis
print("Performing PCA Analysis...")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Find number of components for 90% and 95% variance
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

print(f"Number of components needed for 90% variance: {n_components_90}")
print(f"Number of components needed for 95% variance: {n_components_95}")

# Create PCA transformers for 90% and 95% variance
pca_90 = PCA(n_components=n_components_90)
pca_95 = PCA(n_components=n_components_95)

# Transform the data
X_pca_90 = pca_90.fit_transform(X_scaled)
X_pca_95 = pca_95.fit_transform(X_scaled)

# Split the PCA-transformed data
X_train_90, X_test_90, y_train_90, y_test_90 = train_test_split(
    X_pca_90, y, test_size=0.3, random_state=42
)

X_train_95, X_test_95, y_train_95, y_test_95 = train_test_split(
    X_pca_95, y, test_size=0.3, random_state=42
)

# Plot cumulative variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('pca_variance_plot.png')
plt.close()

# Create results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)
print(f"\nCreating results directory: {results_dir}")

# Function to save model results
def save_model_results(model_name, original_metrics, pca_90_metrics, pca_95_metrics, n_components_90, n_components_95, cumulative_variance_ratio):
    # Create model-specific directory
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Version': ['Original', 'PCA_90%', 'PCA_95%'],
        'Accuracy': [original_metrics['accuracy'], pca_90_metrics['accuracy'], pca_95_metrics['accuracy']],
        'Precision': [original_metrics['precision'], pca_90_metrics['precision'], pca_95_metrics['precision']],
        'Recall': [original_metrics['recall'], pca_90_metrics['recall'], pca_95_metrics['recall']],
        'F1_Score': [original_metrics['f1'], pca_90_metrics['f1'], pca_95_metrics['f1']]
    })
    metrics_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)
    
    # Save detailed results to text file
    with open(os.path.join(model_dir, 'detailed_results.txt'), 'w') as f:
        f.write(f"Detailed Results for {model_name}\n")
        f.write("=" * (len(model_name) + 20) + "\n\n")
        
        # Original Model Results
        f.write("Original Model Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {original_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {original_metrics['precision']:.4f}\n")
        f.write(f"Recall: {original_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {original_metrics['f1']:.4f}\n\n")
        
        # PCA 90% Results
        f.write("PCA 90% Model Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {pca_90_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_90_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_90_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_90_metrics['f1']:.4f}\n\n")
        
        # PCA 95% Results
        f.write("PCA 95% Model Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {pca_95_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_95_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_95_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_95_metrics['f1']:.4f}\n\n")
        
        # Performance Differences
        f.write("Performance Differences:\n")
        f.write("-" * 20 + "\n")
        f.write("PCA 90% - Original:\n")
        f.write(f"Accuracy: {pca_90_metrics['accuracy'] - original_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_90_metrics['precision'] - original_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_90_metrics['recall'] - original_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_90_metrics['f1'] - original_metrics['f1']:.4f}\n\n")
        
        f.write("PCA 95% - Original:\n")
        f.write(f"Accuracy: {pca_95_metrics['accuracy'] - original_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_95_metrics['precision'] - original_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_95_metrics['recall'] - original_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_95_metrics['f1'] - original_metrics['f1']:.4f}\n")
        
        # Add PCA information
        f.write("\nPCA Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"90% variance components: {n_components_90}\n")
        f.write(f"95% variance components: {n_components_95}\n")
        f.write(f"90% variance explained: {cumulative_variance_ratio[n_components_90-1]:.4f}\n")
        f.write(f"95% variance explained: {cumulative_variance_ratio[n_components_95-1]:.4f}\n")
    
    # Create and save model-specific visualizations
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']  # Changed to lowercase to match dictionary keys
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [original_metrics[m] for m in metrics], width, label='Original')
    plt.bar(x, [pca_90_metrics[m] for m in metrics], width, label='PCA 90%')
    plt.bar(x + width, [pca_95_metrics[m] for m in metrics], width, label='PCA 95%')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'{model_name} Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in metrics])  # Capitalize for display
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'performance_comparison.png'))
    plt.close()
    
    print(f"Saved results for {model_name}")

# Define the evaluate_model function
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate a model and return its accuracy and other metrics.
    
    Parameters:
    -----------
    model : estimator object
        The model to evaluate
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    model_name : str
        Name of the model for results storage
        
    Returns:
    --------
    float
        Accuracy score of the model
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return accuracy

# Initialize results dictionary
results = {}

# Define all models to test
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'ElasticNet': ElasticNet(random_state=42)
}

# Train and evaluate all models
print("\nTraining models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Original model
    accuracy_orig = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    print(f"{name} (Original) Accuracy: {accuracy_orig:.4f}")
    
    # PCA 90%
    accuracy_90 = evaluate_model(model, X_train_90, X_test_90, y_train_90, y_test_90, f"PCA_90_{name}")
    print(f"{name} (PCA 90%) Accuracy: {accuracy_90:.4f}")
    
    # PCA 95%
    accuracy_95 = evaluate_model(model, X_train_95, X_test_95, y_train_95, y_test_95, f"PCA_95_{name}")
    print(f"{name} (PCA 95%) Accuracy: {accuracy_95:.4f}")
    
    # Save results immediately after all versions are evaluated
    save_model_results(
        name,
        results[name],
        results[f"PCA_90_{name}"],
        results[f"PCA_95_{name}"],
        n_components_90,
        n_components_95,
        cumulative_variance_ratio
    )

# Create comparison DataFrame
comparison_data = []
for model_name in models.keys():
    original_metrics = results[model_name]
    pca_90_metrics = results[f"PCA_90_{model_name}"]
    pca_95_metrics = results[f"PCA_95_{model_name}"]
    
    comparison_data.append({
        'Model': model_name,
        'Original_Accuracy': original_metrics['accuracy'],
        'PCA_90_Accuracy': pca_90_metrics['accuracy'],
        'PCA_95_Accuracy': pca_95_metrics['accuracy'],
        'Accuracy_Difference_90': pca_90_metrics['accuracy'] - original_metrics['accuracy'],
        'Accuracy_Difference_95': pca_95_metrics['accuracy'] - original_metrics['accuracy'],
        'Original_F1': original_metrics['f1'],
        'PCA_90_F1': pca_90_metrics['f1'],
        'PCA_95_F1': pca_95_metrics['f1'],
        'F1_Difference_90': pca_90_metrics['f1'] - original_metrics['f1'],
        'F1_Difference_95': pca_95_metrics['f1'] - original_metrics['f1']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Accuracy_Difference_90', ascending=False)

# Print comparison results
print("\nModel Comparison (Original vs PCA):")
print("====================================")
print(comparison_df.to_string(index=False))

# Plot comparison
plt.figure(figsize=(15, 8))
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, comparison_df['Original_Accuracy'], width, label='Original')
plt.bar(x + width/2, comparison_df['PCA_90_Accuracy'], width, label='PCA 90%')
plt.bar(x + width/2 + width, comparison_df['PCA_95_Accuracy'], width, label='PCA 95%')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison: Original vs PCA')
plt.xticks(x, comparison_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# After the model comparison section, add code to save results
print("\nSaving results to files...")

# Save comparison DataFrame to CSV
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("Saved comparison results to 'model_comparison_results.csv'")

# Save detailed results to a text file
with open('detailed_model_results.txt', 'w') as f:
    f.write("Detailed Model Performance Analysis\n")
    f.write("=================================\n\n")
    
    # Write PCA Analysis Results
    f.write("PCA Analysis\n")
    f.write("------------\n")
    f.write(f"Number of components needed for 90% variance: {n_components_90}\n")
    f.write(f"Number of components needed for 95% variance: {n_components_95}\n\n")
    
    # Write Model Comparison Results
    f.write("Model Comparison Results\n")
    f.write("----------------------\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")
    
    # Write Detailed Results for Each Model
    f.write("Detailed Results for Each Model\n")
    f.write("-----------------------------\n")
    for model_name in models.keys():
        f.write(f"\n{model_name}:\n")
        f.write("-" * (len(model_name) + 1) + "\n")
        
        # Original Model Results
        f.write("Original Model:\n")
        original_metrics = results[model_name]
        f.write(f"Accuracy: {original_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {original_metrics['precision']:.4f}\n")
        f.write(f"Recall: {original_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {original_metrics['f1']:.4f}\n")
        
        # PCA 90% Model Results
        f.write("\nPCA 90% Model:\n")
        pca_90_metrics = results[f"PCA_90_{model_name}"]
        f.write(f"Accuracy: {pca_90_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_90_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_90_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_90_metrics['f1']:.4f}\n")
        
        # PCA 95% Model Results
        f.write("\nPCA 95% Model:\n")
        pca_95_metrics = results[f"PCA_95_{model_name}"]
        f.write(f"Accuracy: {pca_95_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_95_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_95_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_95_metrics['f1']:.4f}\n")
        
        # Performance Differences
        f.write("\nPerformance Differences:\n")
        f.write(f"Accuracy: {pca_90_metrics['accuracy'] - original_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {pca_90_metrics['precision'] - original_metrics['precision']:.4f}\n")
        f.write(f"Recall: {pca_90_metrics['recall'] - original_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {pca_90_metrics['f1'] - original_metrics['f1']:.4f}\n")
        f.write("\n" + "="*50 + "\n")

print("Saved detailed results to 'detailed_model_results.txt'")

# Save PCA components information
pca_components_df = pd.DataFrame({
    'Component': range(1, len(explained_variance_ratio) + 1),
    'Explained_Variance_Ratio': explained_variance_ratio,
    'Cumulative_Variance_Ratio': cumulative_variance_ratio
})
pca_components_df.to_csv('pca_components_analysis.csv', index=False)
print("Saved PCA components analysis to 'pca_components_analysis.csv'")

# %% [code]
# Step 3: Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Voting algoritmaları uygulanabilir. Henüz denemedim. Grid Search de tabiki :)

# %% [code]
# Evaluate on the test set
y_pred = rf_classifier.predict(X_test)

# %% [code]
# Metrics
overall_accuracy = accuracy_score(y_test, y_pred)
overall_precision = precision_score(y_test, y_pred, average="weighted")
overall_recall = recall_score(y_test, y_pred, average="weighted")
overall_f1 = f1_score(y_test, y_pred, average="weighted")
classification_report_output = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_)

# %% [code]
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
conf_matrix_path = "confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.close()

# %% [code]
# Step 4: Validation using K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_classifier, X, y, cv=kf, scoring='accuracy')

# %% [code]
# Step 5: Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# %% [code]
# Step 6: Generate Detailed Report
print("Detailed Analysis and Classification Report")
print("============================================")
# Dataset Overview
print("1. Dataset Overview")
print("The dataset was preprocessed to standardize numerical features, encode categorical variables, "
                  "and remove multicollinear features such as 'cell_count'.")

print("============================================")


# Model Training and Performance
print("2. Model Training and Performance")
print(
    "Random Forest Classifier achieved the following results on the test set:")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall: {overall_recall:.4f}")
print(f"F1 Score: {overall_f1:.4f}")
print("Classification Report:")
print(classification_report_output)

print("============================================")

# Validation Results
print("3. Validation Results")
print("5-Fold Cross-Validation Accuracy:")
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation: {np.std(cv_scores):.2f}")

print("============================================")

# Feature Importances
print("5. Key Feature Importances")
print("Top features contributing to the classification are:")
for idx, row in feature_importances.head(10).iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")