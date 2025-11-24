# train_models.py
# Trains multiple models (LogisticRegression, DecisionTree, RandomForest, GradientBoosting)
# Evaluates and selects the best model based on ROC AUC, then saves it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import os

from features import prepare_features

MODEL_DIR = '../models'

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs) if probs is not None else None

    print(f"\n{name} metrics:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, AUC: {auc if auc is None else round(auc,3)}")
    print(classification_report(y_test, preds, zero_division=0))
    return {'name': name, 'model': model, 'accuracy': acc, 'precision': prec, 'recall': rec, 'auc': auc}

def train_all(path='../data/sample_noshow.csv'):
    df = pd.read_csv(path)
    X, y = prepare_features(df)
    # simple train-test split (small dataset in sample; in prod use CV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    candidates = [
        ('LogisticRegression', LogisticRegression(solver='liblinear', max_iter=200)),
        ('DecisionTree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]

    results = []
    for name, model in candidates:
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # Choose best by AUC if available, else accuracy
    # Filter models with AUC not None
    aucs = [r for r in results if r['auc'] is not None]
    if aucs:
        best = max(aucs, key=lambda x: x['auc'])
    else:
        best = max(results, key=lambda x: x['accuracy'])

    print(f"\nBest model: {best['name']} (AUC: {best['auc'] if best['auc'] is not None else 'N/A'})")

    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{best['name']}_noshow_model.pkl")
    joblib.dump(best['model'], save_path)
    print("Saved best model to:", save_path)
    return best

if __name__ == "__main__":
    train_all()
