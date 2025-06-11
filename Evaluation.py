import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import time

df = pd.read_csv('path/to/your/file.csv')
X = df.values[:, :-1]
y = df.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

classifiers = [
    RandomForestClassifier(criterion='gini', max_depth=4, n_estimators=20),
    AdaBoostClassifier(learning_rate=1, n_estimators=145),
    DecisionTreeClassifier(criterion='entropy', max_depth=20, max_features=3),
    SVC(C=0.0001, gamma=0.0001, kernel='linear', probability=True),
    XGBClassifier(learning_rate=0.1, n_estimators=80)]

for clf in classifiers:
    T1 = time.time()
    clf.fit(X_train, y_train)
    T2 = time.time()
    name = clf.__class__.__name__

    print("="*30)
    print(name)

    print('****Results****')
    print(T2 - T1)
    train_predictions = clf.predict(X_test)
    y_pre = clf.predict_proba(X_test)
    C = confusion_matrix(y_test, train_predictions)
    print(C)
    acc = accuracy_score(y_test, train_predictions)
    print("ACC: {:.4%}".format(acc))
    pre = precision_score(y_test, train_predictions, average='micro')
    print('Pre: {:.4%}'.format(pre))
    se = recall_score(y_test, train_predictions, average='macro')
    print('SE: {:.4%}'.format(se))
    f1 = f1_score(y_test, train_predictions, average='weighted')
    print('F1: {:.4%}'.format(f1))

    cv_scores = cross_val_score(clf, X_train, y_train, cv=3)
    print("Cross-Validation Accuracy: {:.4%} (+/- {:.4%})".format(cv_scores.mean(), cv_scores.std() * 2))

