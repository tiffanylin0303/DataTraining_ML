import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("processed_ai4i2020.csv")


# Extract features and labels
X = df[["Type_H","Type_L","Type_M","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]]  # Features (input data)
# y = df[["TWF", "HDF", "PWF", "OSF", "RNF"]]
y = df[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

# Ensure no invalid characters in column names
X.columns = X.columns.str.replace('[', '_').str.replace(']', '_').str.replace('<', '_').str.replace('>', '_')
X.columns = X.columns.astype(str)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.3))
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

# Define the hyperparameter grid for KNN
param_grid = {
    'estimator__n_estimators': [x for x in range(10, 100, 10)],  # iterator times
    'estimator__learning_rate': np.arange(0.05, 0.8, 0.1),  
    'estimator__gamma': np.arange(10, 60, 10)
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator = xgb_model,      # Model to tune
    param_grid = param_grid,    # Hyperparameter grid
    cv = 5,                     # 5-fold cross-validation
    scoring ='accuracy',        # Evaluation metric
)

# Train the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Parameters Found:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Data: {accuracy:.4f}")

# 預測成功的比例
print('訓練集: ',xgb_model.score(X_train,y_train))
print('測試集: ',xgb_model.score(X_test,y_test))

# Permutation Importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get feature importance
importance = result.importances_mean

# Create a DataFrame to display feature importance
feat_importances = pd.Series(importance, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
plt.title("Feature Importance (Permutation Importance)")
plt.show()