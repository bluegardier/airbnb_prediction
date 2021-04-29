import fire
import pandas as pd
import numpy as np
from airbnb_prediction import preprocess, config, modelling
import pickle

# Argument only to test Fire's functionality
def features(path: str = '../data/processed/data_train.pickle'):
    """
    Method to split, clean and preprocess data.
    :return:
    """

    #train_save_path = kwargs.get(path, '../data/processed/data_train.pickle')

    # Splits
    df_train, df_test = preprocess.spliting_dataset()

    preprocess.preprocess_data(df_train)
    preprocess.preprocess_data(df_test)

    pickle.dump(df_train, open(path, 'wb'))
    pickle.dump(df_test, open('../data/processed/data_test.pickle', 'wb'))
    print("Generating Datasets for Model Training")


def deploy_model():
    """
    Deploy Model
    :return:
    """

    df = pickle.load(open('../data/processed/data_train.pickle', 'rb'))

    # Model Stage
    print('Starting Model Stage')
    model = modelling.RegressorTrainer(df.drop('id', axis=1), 'price', "testing")

    print('Setting up Pycaret Environment')
    model.start_session()

    print("Training The Model")
    model.train_model()

    print("Finalizing The Model")
    model.finalize_model()

    print('Generated Model Saved at: {}'.format(config.model_path))
    model.save_model("{}/model".format(config.model_path))


def evaluate_model():
    """
    Evaluates model.
    :return:
    """
    df = pickle.load(open('../data/processed/data_test.pickle', 'rb'))

    model = modelling.RegressorTrainer(df.drop('id', axis=1), 'price', "validation")
    model.load_model("{}/model".format(config.model_path))

    prediction = model.predict_model(df, export_metrics=True)
    pickle.dump(prediction, open('../data/processed/prediction.pickle', 'wb'))


def run():
    """
    Run all model pipeline steps sequentially.
    :return:
    """
    features()
    deploy_model()
    evaluate_model()


def cli():
    """ Caller to transform module in a low-level CLI """
    return fire.Fire()


if __name__ == '__main__':
    cli()
