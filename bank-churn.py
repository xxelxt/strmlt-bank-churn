import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
import streamlit as st

def load_data(train_file_path):
    train = pd.read_csv(train_file_path)
    return train

def preprocess_data(train):
    columns_to_drop = ['id', 'CustomerId', 'Surname']
    train.drop(columns=columns_to_drop, inplace=True)
    
    ordinal_encoder = OrdinalEncoder()
    train[['Gender', 'Geography']] = ordinal_encoder.fit_transform(train[['Gender', 'Geography']])
    
    train['AssetRatio'] = train['Balance'] / train['EstimatedSalary']
    train['Interaction'] = train['NumOfProducts'] * (train['IsActiveMember'] + 0.2)
    
    return train, ordinal_encoder

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    train_auc_score = roc_auc_score(y_train, train_pred_proba)
    test_auc_score = roc_auc_score(y_test, test_pred_proba)
    
    evaluation_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'Accuracy': [train_accuracy, test_accuracy],
        'AUC Score': [train_auc_score, test_auc_score]
    })
    
    return model, evaluation_df

def ensemble_predictions(models, X):
    predictions = np.zeros((X.shape[0], len(models)))
    
    for i, model in enumerate(models):
        predictions[:, i] = model.predict_proba(X)[:, 1]
        
    return predictions

def main():
    train_file_path = 'train.csv'
    
    train = load_data(train_file_path)
    st.write("### Dữ liệu train:")
    st.write(train)

    train, ordinal_encoder = preprocess_data(train)

    st.write("### Dữ liệu sau khi xử lý:")
    st.write(train)
    
    X = train.drop('Exited', axis=1)
    y = train['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    models = []
    evaluation_results = {}

    rf_model = RandomForestClassifier(n_estimators=1500, random_state=42, max_depth=10)
    rf_model, rf_evaluation = evaluate_model(rf_model, X_train, y_train, X_test, y_test)
    models.append(rf_model)
    evaluation_results['RandomForestClassifier'] = rf_evaluation
    
    xgboost = XGBClassifier(n_estimators=1500, learning_rate=0.005, n_jobs=-1, random_state=42, eval_metric=['auc'])
    xgboost, xgboost_evaluation = evaluate_model(xgboost, X_train, y_train, X_test, y_test)
    models.append(xgboost)
    evaluation_results['XGBClassifier'] = xgboost_evaluation
    
    catboost = CatBoostClassifier(eval_metric='AUC', learning_rate=0.01, iterations=10000, random_seed=3)
    catboost, catboost_evaluation = evaluate_model(catboost, X_train, y_train, X_test, y_test)
    models.append(catboost)
    evaluation_results['CatBoostClassifier'] = catboost_evaluation
    
    lgbm = LGBMClassifier(learning_rate=0.01, n_estimators=1500, max_depth=16, num_leaves=16, random_state=42)
    lgbm, lgbm_evaluation = evaluate_model(lgbm, X_train, y_train, X_test, y_test)
    models.append(lgbm)
    evaluation_results['LGBMClassifier'] = lgbm_evaluation
    
    st.write("### Đánh giá các mô hình:")
    for model_name, evaluation_df in evaluation_results.items():
        st.write(f"- {model_name}:")
        st.write(evaluation_df)
    
    ensemble_predictions_test = ensemble_predictions(models, X_test)
    combined_predictions_test = np.mean(ensemble_predictions_test, axis=1)
    
    ensemble_accuracy = accuracy_score(y_test, np.round(combined_predictions_test))
    ensemble_auc_score = roc_auc_score(y_test, combined_predictions_test)
    
    st.write("### Đánh giá mô hình kết hợp:")
    st.write(f"Accuracy: {ensemble_accuracy}")
    st.write(f"AUC Score: {ensemble_auc_score}")
    
    ensemble_model_path = 'ensemble_model.pkl'
    with open(ensemble_model_path, 'wb') as f:
        pickle.dump((models, ordinal_encoder), f)

if __name__ == "__main__":
    main()