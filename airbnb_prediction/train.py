import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv("../data/cleaned/df_to_model.csv")

    X = df.drop(['price', 'log_price'], axis=1)
    y = df['price']

    X_train, X_test, \
    y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    n_estimators = int(sys.argv[1])

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Random Forest Regressor model, n_estimators =  ):".format(n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")
