import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

class UiConn:
    def __init__(self):
        self.tracking_uri = mlflow.get_tracking_uri()

    def create_ui_session(self, port=5555):
        mlflow.set_tracking_uri(self.tracking_uri)
        get_ipython() \
            .system_raw(
            "mlflow ui --backend-store-uri {} --port {} &"
                .format(self.tracking_uri, port)
        )
        print('Access for UI at: http://127.0.0.1:{}'.format(port))

    @staticmethod
    def terminate_ui_session():
        get_ipython() \
            .system_raw(
            "pkill -f gunicorn"
        )
        print('MLflow UI session terminated')


class TrainerReg:
    def __init__(self, estimator, params={}):
        self._model = estimator(**params)
        self._params = params

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

    def mlflow_run(self, df, target, model_name, r_name="default_experiment", log_price=False):
        with mlflow.start_run(run_name=r_name) as run:
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            X = df.drop(target, axis=1)
            y = df[target]

            X_train, X_test, \
            y_train, y_test = \
                train_test_split(X, y, test_size=0.2, random_state=0)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            self._model.fit(X_train, y_train)
            y_pred = self._model.predict(X_test)

            mlflow.sklearn.log_model(self._model, model_name)
            mlflow.log_params(self._params)

            if log_price:
                y_test = np.exp(y_test)
                y_pred = np.exp(y_pred)

            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rmse)
            print('R2                     :', r2)

            return experimentID, runID