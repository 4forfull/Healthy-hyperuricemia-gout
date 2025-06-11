import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

df = pd.read_csv('path/to/your/file.csv')
X = df.values[:, :-1]
y = df.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

def run_grid_search(model, param_grid, model_name, cv=3):
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"\n===== {model_name} Best Results =====")
    print("Best Score:     ", grid.best_score_)
    print("Best Params:    ", grid.best_params_)
    print("Best Estimator: ", grid.best_estimator_)
    return grid


# Random Forest
def random_forest_search():
    param_grid = {
        'n_estimators': [3, 5, 7, 10, 15, 20, 50, 60, 80, 100, 110, 120, 130, 140],
        'max_depth': range(3, 20),
        'criterion': ['gini', 'entropy']
    }
    return run_grid_search(RandomForestClassifier(), param_grid, "Random Forest")


# Decision Tree
def decision_tree_search():
    param_grid = {
        'max_depth': range(1, 30),
        'max_features': [3, 5, 7, 10, 15, 20, 25, 30, 35, 40],
        'criterion': ["entropy", "gini"]
    }
    return run_grid_search(DecisionTreeClassifier(), param_grid, "Decision Tree")


# AdaBoost
def adaboost_search():
    param_grid = {
        'n_estimators': [3, 5, 7, 10, 15, 20, 50, 60, 80, 100, 120, 130, 140, 150, 160, 170, 180],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    return run_grid_search(AdaBoostClassifier(), param_grid, "AdaBoost")


# SVM
def svm_search():
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    return run_grid_search(SVC(kernel='linear', probability=True), param_grid, "SVM")


# XGBoost
def xgboost_search():
    param_grid = {
        'n_estimators': [3, 5, 7, 10, 15, 20, 50, 60, 80, 100, 120, 130, 140, 150, 160, 170, 180],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    return run_grid_search(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grid, "XGBoost", cv=5)

if __name__ == "__main__":
    random_forest_search()
    decision_tree_search()
    adaboost_search()
    svm_search()
    xgboost_search()