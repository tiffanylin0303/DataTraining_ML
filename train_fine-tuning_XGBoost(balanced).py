import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# Load the dataset
df = pd.read_csv("processed_ai4i2020.csv")

df = df[~(df[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1) > 1)]
df["OK"] = df["Machine failure"].apply(lambda x: 0 if x == 1 else 1)
df.loc[df["RNF"] == 1, "OK"] = 0

# Extract features and labels
X = df[["Type_H","Type_L","Type_M","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]]  # Features (input data)
y = df[["OK", "TWF", "HDF", "PWF", "OSF", "RNF"]]
y = np.argmax(y.values, axis=1)

target_names = ["OK", "TWF", "HDF", "PWF", "OSF", "RNF"]

# Ensure no invalid characters in column names
X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_').str.replace('>', '_')
X.columns = X.columns.astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.3, class_weight='balanced')
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train 
)

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.3)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred = xgb_model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

# Define the hyperparameter grid for KNN
param_grid = {
    'n_estimators':  [10, 50, 100, 150],
    'learning_rate':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator = xgb_model,      # Model to tune
    param_grid = param_grid,    # Hyperparameter grid
    cv = 5,                     # 5-fold cross-validation
    scoring ='accuracy',        # Evaluation metric
    n_jobs=-1, 
)

# Train the model with GridSearchCV
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1s = f1_score(y_test, y_pred, average='weighted')
print("XGBoost")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Recall: {recall:.2%}")
print(f"Precision: {precision:.2%}")
print(f"F1-Score: {f1s:.2%}")

# Confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=target_names)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)  # normalize or format if needed
plt.title("Confusion Matrix")
plt.show()
# Permutation Importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get feature importance
importance = result.importances_mean

# Create a DataFrame to display feature importance
feat_importances = pd.Series(importance, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
plt.title("Feature Importance (XGBoost model)")
plt.show()