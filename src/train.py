import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from src.data import load_data, split_data
from src.evaluate import evaluasi


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [200, 500],
        "max_leaf_nodes": [31, 63],
    }
    grid = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Parameter terbaik:", grid.best_params_)
    print(f"CV MAE terbaik: {-grid.best_score_:.3f}")
    evaluasi(grid.best_estimator_, X_test, y_test, "Gradient Boosting (tuned)")

    joblib.dump(grid.best_estimator_, "models/gradient_boosting.joblib")


if __name__ == "__main__":
    main()