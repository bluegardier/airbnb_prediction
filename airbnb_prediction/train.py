import pandas as pd
import numpy as np
from airbnb_prediction import preprocess, config, modelling
import pickle

if __name__ == '__main__':
    # Preprocess Step
    df = pd.read_csv('{}/listings_train.csv'.format(config.data_dir_raw))

    preprocess.preprocess_data(df)
    pickle.dump(df, open('../data/processed/model_data.pickle', 'wb'))

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

    print('Model Performance:')
    print(round(model.metrics.loc['Mean'], 2))
