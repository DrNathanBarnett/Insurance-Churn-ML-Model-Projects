# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:59:11 2025

@author: Nathan
"""

conda install seaborn

import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as temp_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import shap


df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.info()


label_encoder = LabelEncoder()
df['pricesensitive_cheaper_to_competitor_min'] = label_encoder.fit_transform(df['pricesensitive_cheaper_to_competitor_min'])
#df = pd.get_dummies(df, columns = ['pricesensitive_cheaper_to_competitor_min'], drop_first=True)


df.info()

features = ['pricesensitive_number_of_quotes', 
    'pricesensitive_ratio_price_increase_over_original',
    'loyalty_termno',  
    'affluence_seifa_socioeconomic',
    'affluence_seifa_economic',
    'pricesensitive_ratio_price_to_competitor_min',
    'pricesensitive_cheaper_to_competitor_min' ,
    'pricesensitive_partner_name_aggregator',
    'affluence_original_premium',
    'loyalty_insurance'
]

X = df[features]
y = df['affluence_payment_yearly']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestClassifier(n_estimators= 100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(df.columns)
print(conf_matrix)
print(class_report)
print(accuracy)

model_rf_balanced = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    # Add the class weight parameter
    class_weight='balanced'
)
model_rf_balanced.fit(X_train, y_train)

# Rerun prediction and evaluation
y_pred_rf_balanced = model_rf_balanced.predict(X_test)
print("--- Random Forest with Balanced Weights ---")
print(confusion_matrix(y_test, y_pred_rf_balanced))
print(classification_report(y_test, y_pred_rf_balanced))

importances = model.feature_importances_
indicies = np.argsort(importances)[::-1]
names = [features[i] for i in indicies]

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), importances[indicies])
plt.yticks(range(X.shape[1]), names)
plt.show()

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

y_pred_log_reg = log_reg.predict(X_test)

conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(conf_matrix_log_reg, class_report_log_reg, accuracy_log_reg)


from sklearn.svm import SVC

svm_model = SVC(kernel = 'linear', random_state = 42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(conf_matrix_svm, class_report_svm, accuracy_svm)


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_kmm)

print(conf_matrix_knn, class_report_knn, accuracy_knn)


from sklearn.ensemble import GradientBoostingClassifier

gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)

y_pred_gbm = gbm_model.predict(X_test)

conf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)
class_report_gbm = classification_report(y_test, y_pred_gbm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)

print(conf_matrix_gbm, class_report_gbm, accuracy_gbm)







