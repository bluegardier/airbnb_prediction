import pandas as pd
import pycaret.regression as pcr
from airbnb_prediction import config


class RegressorTrainer:
    def __init__(self, df: pd.DataFrame, target: str, exp_name: str, session_id: int = 16):
        """
        Initialize classe objects.
        :param df: Cleaned dataframe for model ingestion.
        :param target: Target variable.
        :param exp_name: Model experiment name, for mlflow tracking purposes.
        :param session_id: experiment's random state.
        """
        self.df = df
        self.target = target
        self.exp_name = exp_name
        self.session_id = session_id

    def start_session(self):
        """
        Starts pycaret environment.
        :return: None
        """
        pcr.setup(
            data=self.df,
            target=self.target,
            experiment_name=self.exp_name,
            categorical_features=config.pycaret_categorical_features,
            numeric_features=config.pycaret_numerical_features,
            session_id=self.session_id,
            normalize=True,
            silent=True,
            verbose=False
        )

    def train_model(self):
        """
        Train and optimize model.
        :return: Final model.
        """
        print('Training LightGBM: Step 1/3')
        lightgbm = pcr.create_model('lightgbm',
                                verbose=False)

        print('Training Tuned LightGBM, Optimize = RMSE: Step 2/3')
        tuned_lightgbm = pcr.tune_model(lightgbm, optimize='RMSE',
                                    verbose=False)

        print('Training Ensemble LightGBM: Step 3/3')
        self.model = pcr.ensemble_model(tuned_lightgbm,
                                    verbose=False)
        self.metrics = pcr.pull()

    def finalize_model(self):
        """
        This function fits the estimator onto the complete
        dataset passed during the setup() stage.
        The purpose of this function is to
        prepare for final model deployment after experimentation.
        Description taken from pycaret/classification.
        :return: Trained model object fitted on complete dataset.
        """
        pcr.finalize_model(self.model)

    def save_model(self, path: str):
        """
        Saves model to path.
        :param path: Path to save model.
        :return: None
        """
        pcr.save_model(self.model, path)

    def predict_model(self, data: pd.DataFrame):
        """
        Predicts model on unseen data.
        :param data: a new unseen dataframe.
        :return: float. Prediction on unseen data.
        """
        prediction = pcr.predict_model(estimator=self.model, data=data)
        return prediction['Label'][0]

    @property
    def get_model(self):
        return self.model
