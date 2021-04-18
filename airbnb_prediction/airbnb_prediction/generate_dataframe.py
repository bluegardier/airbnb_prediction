import pandas as pd
import numpy as np
import pickle

from airbnb_prediction import eda_utils, objects


def generate_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Final Configuration do generate dataframe model.
    :param dataframe:
    :return:
    """
    dataframe['price'] = dataframe['price'] \
        .apply(lambda x: eda_utils.convert_price_to_int(x))

    dataframe['days_since_host'] = \
        (pd.to_datetime('today') - pd.to_datetime(dataframe['host_since'])).dt.days

    dataframe['bathroom_text_clean'] = \
        eda_utils.extract_numbers(dataframe, 'bathrooms_text', fillna=True)

    dataframe['bathrooms'] = np.where(dataframe['bathroom_text_clean'].isnull() == False,
                                      (dataframe['bathroom_text_clean']).astype(float).apply(np.floor), 0)

    dataframe['half_bath'] = \
        np.where(dataframe['bathroom_text_clean'].str.isalnum() == False, 'yes', 'no')

    dataframe['delta_nights'] = \
        eda_utils.creating_delta_variable(dataframe, 'minimum_nights', 'maximum_nights')

    dataframe['delta_date_reviews'] = \
        eda_utils.creating_delta_date_variable(dataframe, 'first_review', 'last_review')

    dataframe['mean_reviews'] = \
        dataframe['number_of_reviews'] / (dataframe['number_of_reviews'].fillna(0) + 1)

    dataframe['regiao'] = eda_utils.creating_zones(dataframe)

    dataframe['property_type_refactor'] = \
        eda_utils.creating_property_type_refactor(dataframe)

    dataframe['is_host_rj'] = eda_utils.creating_host_location(df)

    eda_utils.count_characters_variables(dataframe, objects.string_variables)
    return dataframe


if __name__ == '__main__':
    df = pd.read_csv('../../data/raw/listings.csv')
    generate_dataframe(df)

    df = eda_utils.dropping_empty_columns(df)

    fillna_dict = {
        'host_response_time': 'no_info',
        'host_is_superhost': df['host_is_superhost'].mode()[0],
        'bedrooms': df['bedrooms'].mode()[0],
        'beds': df['beds'].mode()[0],
        'days_since_host': df['days_since_host'].mode()[0]
    }

    df.fillna(fillna_dict, inplace=True)

    df.drop(objects.to_drop, axis=1, inplace=True)

    pickle.dump(df, open('../../data/processed/model_data.pickle', 'wb'))
