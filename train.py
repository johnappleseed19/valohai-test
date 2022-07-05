import logging
import sys

# import mlflow
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        data = pd.read_csv("./data/Pilze.csv", sep=";")
    except Exception as e:
        logger.exception(
            "Unable to read training & test CSV. Error: %s", e
        )

    # prepare data
    del data["gill-attachment"]
    del data["ring-number"]
    del data["veil-color"]
    del data["veil-type"]

    target = np.unique(data["class"])

    x = data.drop(columns="class")
    y = data["class"]

    # encode ordinal data
    enc = OrdinalEncoder()
    x_transformed = enc.fit_transform(x)

    feature_names = x.columns
    labels = y.unique()

    # split the dataset
    train_x, test_x, train_y, test_lab = train_test_split(
        x_transformed, y, test_size=0.2)

    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # with mlflow.start_run() as run:
    #     clf = DecisionTreeClassifier(max_depth=max_depth)
    #     clf.fit(train_x, train_y)
    #     tree_rules = export_text(clf, feature_names=list(feature_names))

    #     print(f"Decision Tree Classifier (max_depth={max_depth}):")

    #     mlflow.log_param("max_depth", max_depth)

    #     # predict class based on test values
    #     test_pred_decision_tree = clf.predict(test_x)

    #     accuracy = metrics.accuracy_score(test_lab, test_pred_decision_tree)
    #     mlflow.log_metric("accuracy", accuracy)

    #     confusion_matrix = metrics.confusion_matrix(
    #         test_lab,  test_pred_decision_tree)

    #     tn, fp, fn, tp = confusion_matrix.ravel()
    #     mlflow.log_metric("true_negatives", tn)
    #     mlflow.log_metric("false_positives", fp)
    #     mlflow.log_metric("false_negatives", fn)
    #     mlflow.log_metric("true_positives", tp)
