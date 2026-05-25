import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

    return metrics



def compare_models(results_dict):
    """
    Compara métricas de múltiples modelos.
    """

    return pd.DataFrame(results_dict).T



def generate_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)



def get_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)