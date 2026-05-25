from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



def optimize_random_forest(X_train, y_train):
    """
    Optimización de Random Forest usando GridSearchCV.
    """

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_



def optimize_svm(X_train, y_train):
    """
    Optimización de SVM usando GridSearchCV.
    """

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    }

    model = SVC()

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_