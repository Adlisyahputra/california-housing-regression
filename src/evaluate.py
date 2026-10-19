from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluasi(model, X_test, y_test, nama="Model"):
    pred = model.predict(X_test)
    print(nama)
    print(f"  MAE  : {mean_absolute_error(y_test, pred):.3f}")
    print(f"  RMSE : {root_mean_squared_error(y_test, pred):.3f}")
    print(f"  R2   : {r2_score(y_test, pred):.3f}")