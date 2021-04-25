import pandas as pd
import numpy as np
from airbnb_prediction import preprocess, config, modelling
import pickle

if __name__ == '__main__':
    df = pd.read_csv('{}/listings_train.csv'.format(config.data_dir_raw))
    preprocess.preprocess_data(df)

    df = preprocess.dropping_empty_columns(df)

    fillna_dict = {
        'host_response_time': 'no_info',
        'host_is_superhost': df['host_is_superhost'].mode()[0],
        'bedrooms': df['bedrooms'].mode()[0],
        'beds': df['beds'].mode()[0],
        'days_since_host': df['days_since_host'].mode()[0]
    }

    df.fillna(fillna_dict, inplace=True)

    df.drop(config.to_drop, axis=1, inplace=True)
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




