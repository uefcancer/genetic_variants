import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('train_df.csv', delimiter=',')
exclude_columns = ['SampleID', 'CaseControl']

# Calculate mode of each column in the training set
modes = {}
for column in df_train.columns:
    if column not in exclude_columns:
        modes[column] = df_train[column].mode()[0]

# Replace -1 values with the mode of each column in the training set
for column in df_train.columns:
    if column not in exclude_columns:
        df_train[column] = df_train[column].replace(-1, modes[column])

# Identify columns where more than 10% of the values are -1
columns_to_remove = [column for column in df_train.columns if (df_train[column] == -1).mean() > 0.1]

# Remove these columns from the training set
df_train = df_train.drop(columns_to_remove, axis=1)

df_test = pd.read_csv('test_df.csv', delimiter=',')

# Replace -1 values with the mode of each column in the test set
for column in df_test.columns:
    if column not in exclude_columns and column in modes:
        df_test[column] = df_test[column].replace(-1, modes[column])

df_test = df_test.drop(columns_to_remove, axis=1)

ID = df_train['SampleID'].tolist()
y_train = df_train['CaseControl'].tolist()
X_train = df_train.drop(exclude_columns, axis=1)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

m = 6  # windows size
W = 1  # increment
S = 50 # top features

# initial model to get feature importance
model = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.01, objective='binary:logistic')
model.fit(X_train, y_train)
importance = model.get_booster().get_score(importance_type='weight')

sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True) # Sort features by importance
# Initialize top and bottom feature lists
top_features = [f[0] for f in sorted_features[:m]] 
bottom_features = [f[0] for f in sorted_features[-m:]]

model_top = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.01, objective='binary:logistic')
model_bottom = XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.01, objective='binary:logistic')

while set(top_features).intersection(bottom_features) == set():
    # Fit model on top and bottom features separately
    model_top.fit(X_train[top_features], y_train)
    model_bottom.fit(X_train[bottom_features], y_train)
    
    # Calculate feature importance
    importance_top = model_top.get_booster().get_score(importance_type='weight')
    importance_bottom = model_bottom.get_booster().get_score(importance_type='weight')
    
    # Sort top and bottom features by importance
    top_features = sorted(top_features, key=lambda x: importance_top.get(x, 0), reverse=True)
    bottom_features = sorted(bottom_features, key=lambda x: importance_bottom.get(x, 0), reverse=True)
    
    # Substitute features
    top_features[-1], bottom_features[0] = bottom_features[0], top_features[-1]
    
    m += W
    top_features = [f[0] for f in sorted_features[:m]]
    bottom_features = [f[0] for f in sorted_features[-m:]]


selected_features = [feature[0:] for feature in top_features[:S]] # Select top S features for prediction
X_train = X_train[selected_features]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = SVC(probability=True, random_state=3, kernel='sigmoid', C=1.5, class_weight='balanced')
model.fit(X_train_scaled, y_train)


# Prediction
X_test = df_test[selected_features]
X_test_scaled = scaler.transform(X_test)

y_true = df_test['CaseControl'].tolist()
actual_distribution = Counter(y_true)

y_pred = model.predict(X_test_scaled)
predicted_distribution = Counter(y_pred)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)


print(f"Actual distribution: {actual_distribution}")
print(f"Predicted distribution: {predicted_distribution}")
print('Precision:', precision)
print('Recall:', recall)
print('AUC:', auc)
print('Average Precision:', average_precision)
print('Accuracy:', accuracy)