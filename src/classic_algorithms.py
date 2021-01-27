import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def load_data(test=False):
    if test:
        pass

    classic = pd.read_csv('../input/train_classic_features.csv')
    exotic = pd.read_csv('../input/train_exotic_features.csv')

    train_df = pd.merge(classic, exotic, how="outer")

    labels_to_drop = ["Unnamed: 0", "qid", "question_text", "target", "processed", "stemmed", "wordcloud",
                      "lower_question_text"]

    y = train_df["target"]
    X = train_df.drop(labels_to_drop, axis=1)

    return X, y


def log_reg(X, y):
    classifier = LogisticRegression(max_iter=10000, n_jobs=-1)
    scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    print('F1 for Logistic Regression: ', scores.mean())


def random_forest(X, y):
    classifier = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
    scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    print('F1 for Random Forest: ', scores.mean())


def xgb(X, y):
    classifier = XGBClassifier(scale_pos_weight=100, n_jobs=8)
    scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    print('F1 for XGBoost: ', scores.mean())


def run():
    X, y = load_data()

    # log_reg(X, y)
    # random_forest(X, y)
    xgb(X, y)


if __name__ == "__main__":
    run()
