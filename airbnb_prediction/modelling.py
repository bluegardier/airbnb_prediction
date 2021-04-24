import shap
import pickle
from pycaret.regression import *
from airbnb_prediction import objects


class RegressorTrainer:
    def __init__(self, df: pd.DataFrame, target: str, exp_name: str, session_id: int = 16):
        self.df = df
        self.target = target
        self.exp_name = exp_name
        self.session_id = session_id

    def start_session(self):
        setup(
            data=self.df,
            target=self.target,
            experiment_name=self.exp_name,
            categorical_features=objects.pycaret_categorical_features,
            numeric_features=objects.pycaret_numerical_features,
            session_id=self.session_id,
            normalize=True,
            silent=True,
            verbose=False
        )

    def train_model(self):
        print('Training LightGBM: Step 1/3')
        lightgbm = create_model('lightgbm')
        print('Training Tuned LightGBM, Optimize = RMSE: Step 2/3')
        tuned_lightgbm = tune_model(lightgbm, optimize='RMSE')
        print('Training Ensemble LightGBM: Step 3/3')
        self.model = ensemble_model(tuned_lightgbm)

    def finalize_model(self):
        finalize_model(self.model)

    def save_model(self, path: str):
        pickle.dump(self.model, open(path, 'wb'))

    def predict_model(self, data: pd.DataFrame):
        prediction = predict_model(estimator=self.model, data=data)
        return prediction['Label'][0]

    @property
    def get_model(self):
        return self.model