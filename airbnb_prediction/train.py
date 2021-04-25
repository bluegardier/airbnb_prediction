import pandas as pd
import numpy as np
from airbnb_prediction import preprocess, config, modelling
import pickle


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Final Configuration do generate dataframe model.
    :param dataframe:
    :return:
    """
    dataframe['price'] = dataframe['price'] \
        .apply(lambda x: preprocess.convert_price_to_int(x))

    dataframe['days_since_host'] = \
        (pd.to_datetime('today') - pd.to_datetime(dataframe['host_since'])).dt.days

    dataframe['bathroom_text_clean'] = \
        preprocess.extract_numbers(dataframe, 'bathrooms_text', fillna=True)

    dataframe['bathrooms'] = np.where(dataframe['bathroom_text_clean'].isnull() == False,
                                      (dataframe['bathroom_text_clean']).astype(float).apply(np.floor), 0)

    dataframe['half_bath'] = \
        np.where(dataframe['bathroom_text_clean'].str.isalnum() == False, 'yes', 'no')

    dataframe['delta_nights'] = \
        preprocess.creating_delta_variable(dataframe, 'minimum_nights', 'maximum_nights')

    dataframe['delta_date_reviews'] = \
        preprocess.creating_delta_date_variable(dataframe, 'first_review', 'last_review')

    dataframe['mean_reviews'] = \
        dataframe['number_of_reviews'] / (dataframe['number_of_reviews'].fillna(0) + 1)

    dataframe['regiao'] = preprocess.creating_zones(dataframe)

    dataframe['property_type_refactor'] = \
        preprocess.creating_property_type_refactor(dataframe)

    dataframe['is_host_rj'] = preprocess.creating_host_location(df)

    preprocess.count_characters_variables(dataframe, config.string_variables)
    return dataframe


if __name__ == '__main__':
    df = pd.read_csv('{}/listings.csv'.format(config.data_dir_raw))
    preprocess_data(df)

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




