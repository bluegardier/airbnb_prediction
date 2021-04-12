import mlflow
import mlflow.sklearn

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

    def terminate_ui_session(self):
        get_ipython() \
            .system_raw(
            "pkill -f gunicorn"
        )
