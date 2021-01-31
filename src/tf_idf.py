import pandas as pd
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


def get_data():
    df = pd.read_csv('../input/train_exotic_features.csv')
    X_train, X_test, y_train, y_test = train_test_split(df['stemmed'], df['target'], test_size=0.2)

    return X_train, X_test, y_train, y_test


def vectorize_data(x_train, x_test):
    vectorizer = TfidfVectorizer()
    vectorized_train = vectorizer.fit_transform(x_train)
    vectorized_test = vectorizer.transform(x_test)

    return vectorized_train, vectorized_test


def get_classifier(name):
    if name == "logreg":
        classifier = LogisticRegression(max_iter=10000, n_jobs=-1)
    elif name == "bayes":
        classifier = MultinomialNB(alpha=0.01, fit_prior=True)
    elif name == "rf":
        classifier = RandomForestClassifier(criterion='gini', n_estimators=100, n_jobs=-1)
    elif name == "xgb":
        classifier = XGBClassifier(
            scale_pos_weight=100,
            eta=0.01,
            max_depth=10,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="auc"
        )
    else:
        raise ValueError("No classifier specified")

    return classifier


def run():
    x_train, x_test, y_train, y_test = get_data()
    x_vec_train, x_vec_test = vectorize_data(x_train, x_test)

    for classifier_name in ["logreg", "bayes", "rf", "xgb"]:
        classifier = get_classifier(classifier_name)

        classifier.fit(x_vec_train, y_train)
        y_predicted = classifier.predict(x_vec_test)

        f1 = metrics.f1_score(y_test, y_predicted)

        print(f"Classifier: {classifier_name}, F-score: {f1}")


if __name__ == "__main__":
    run()
