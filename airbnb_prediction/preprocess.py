import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from airbnb_prediction import config


def plot_configuration(x: float = 11.7, y: float = 8.27) -> None:
    """
    Customizable plot size configuration for matplotlib/seaborn plots.
    Run this function right before your seaborn plots.
    :param x: x-axis size.
    :param y: y-axis size.
    :return: None
    """

    a4_dims = (x, y)
    fig, ax = plt.subplots(figsize=a4_dims)


def plot_missing_values(df: pd.DataFrame):
    """
    Plot missing values for entire dataframe.
    :param df: a pd.DataFrame of interest.
    :return: Plot with missing values for each variable.
    """
    prop_missings = (df.isna().sum().sort_values(ascending=False) / df.shape[0])
    f, _ = plt.subplots(figsize=(20, 10))
    plt.xticks(rotation='90')
    sns.barplot(x=prop_missings.index, y=prop_missings)
    plt.xlabel('Variables', fontsize=15)
    plt.ylabel('% of Missings', fontsize=15)
    plt.title('% of Missings for Variable', fontsize=15)


def convert_price_to_int(price: str) -> int:
    """
    Convert a string number to integer.

    :param price: str
    Price value with str type.

    :return: integer
    Price as int type.
    """
    price = int(price[1:-3].replace(",", ""))
    return price


def treat_string_variables(dataframe: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Treat string variables for further applications.
    :param dataframe: pd.DataFrame
    :param variables: list
    List of string variables.

    :return: pd.DataFrame
    """
    for variable in variables:
        dataframe[variable] \
            .fillna(" ", inplace=True)

        dataframe[variable] = \
            dataframe[variable] \
                .apply(lambda x: x.replace(" ", ""))

    return dataframe


def count_characters_variables(dataframe: pd.DataFrame, variables: list) -> None:
    """
    Create count variables from variables list parameter.
    :param dataframe: pd.DataFrame
    Original Dataframe.
    :param variables: list
    List of string variables.

    :return: pd.DataFrame
    """

    treat_string_variables(dataframe, variables)
    for variable in variables:
        dataframe['count_{}'.format(variable)] = \
            dataframe[variable]. \
                apply(lambda x: len(x))

    return dataframe


def extract_numbers(df: pd.DataFrame, variable: str, fillna=True) -> None:
    """
    Extract numbers from strings.
    :param fillna: Fillna with zero if true.
    :param df: pd.DataFrame with the string variable.
    :param variable: String variable containing numbers.
    :return: pd.Series
    """
    if fillna:
        df[variable] = df[variable].fillna(0)

    numbers = np.array(
        df[variable]
            .str
            .extract(
            '([0-9][,.]*[0-9]*)'
        )
    )

    return numbers


def creating_zones(df: pd.DataFrame) -> None:
    """
    Cluster neighborhoods into zones.
    :param df:
    :return:
    """

    regiao = np.where(df['neighbourhood_cleansed'].isin(config.centro), 'centro',
                      np.where(df['neighbourhood_cleansed'].isin(config.zona_sul), 'zona_sul',
                               np.where(df['neighbourhood_cleansed'].isin(config.zona_norte), 'zona_norte',
                                        np.where(df['neighbourhood_cleansed'].isin(config.zona_norte),
                                                 'zona_norte',
                                                 np.where(df['neighbourhood_cleansed'].isin(config.zona_oeste),
                                                          'zona_oeste', 'not_found')
                                                 )
                                        )
                               )
                      )

    return regiao


def creating_host_location(df: pd.DataFrame) -> None:
    """
    Flag indicating if host is in RJ.
    :param df:
    :return:
    """

    regiao_host = np.where(df['host_neighbourhood'].isin(config.centro) |
                           df['host_neighbourhood'].isin(config.zona_sul) |
                           df['host_neighbourhood'].isin(config.zona_norte) |
                           df['host_neighbourhood'].isin(config.zona_oeste), 'yes', 'no')
    return regiao_host


def creating_property_type_refactor(df: pd.DataFrame) -> None:
    """
    Refactor the variable into three categories.
    :param df: the pd.DataFrame
    :return: pd.Series
    """

    df['property_type_refactor'] = np.where(
        (df['property_type'] == 'Private room in apartment') | (df['property_type'] == 'Private room in house'),
        'private_room',
        np.where(df['property_type'] == 'Entire apartment', 'apartment', 'other')
    )

    return df['property_type_refactor']


def creating_delta_variable(df: pd.DataFrame, minimum_variable: str, maximum_variable: str) -> None:
    """
    Creates a variable holding the diference between two numeric variables.
    :param df: pd.DataFrame
    :param minimum_variable: The lower variable.
    :param maximum_variable: The higher variable.
    :return: pd.Series with delta between two variables.
    """

    delta = df[maximum_variable] - df[minimum_variable]

    return delta


def creating_delta_date_variable(df: pd.DataFrame, minimum_date: str, maximum_date: str) -> None:
    """
    Creates a variable holding the diference between two datetime variables.
    :param df: pd.DataFrame
    :param minimum_variable: The lower date variable.
    :param maximum_variable: The higher date variable.
    :return: pd.Series with delta between two date variables.
    """

    delta_date = (pd.to_datetime(df[maximum_date]) -
                  pd.to_datetime(df[minimum_date])).dt.days

    return delta_date


def dropping_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns with no entries.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    prop_missings = (df.isna().sum().sort_values(ascending=False) / df.shape[0])
    to_drop_full_nan = prop_missings[prop_missings == 1].index.to_list()
    df.drop(to_drop_full_nan, axis=1, inplace=True)
    return df


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Final Configuration do generate dataframe model.
    :param dataframe:
    :return:
    """
    dataframe['price'] = dataframe['price'] \
        .apply(lambda x: convert_price_to_int(x))

    dataframe['days_since_host'] = \
        (pd.to_datetime('today') - pd.to_datetime(dataframe['host_since'])).dt.days

    dataframe['bathroom_text_clean'] = \
        extract_numbers(dataframe, 'bathrooms_text', fillna=True)

    dataframe['bathrooms'] = np.where(dataframe['bathroom_text_clean'].isnull() == False,
                                      (dataframe['bathroom_text_clean']).astype(float).apply(np.floor), 0)

    dataframe['half_bath'] = \
        np.where(dataframe['bathroom_text_clean'].str.isalnum() == False, 'yes', 'no')

    dataframe['delta_nights'] = \
        creating_delta_variable(dataframe, 'minimum_nights', 'maximum_nights')

    dataframe['delta_date_reviews'] = \
        creating_delta_date_variable(dataframe, 'first_review', 'last_review')

    dataframe['mean_reviews'] = \
        dataframe['number_of_reviews'] / (dataframe['number_of_reviews'].fillna(0) + 1)

    dataframe['regiao'] = creating_zones(dataframe)

    dataframe['property_type_refactor'] = \
        creating_property_type_refactor(dataframe)

    dataframe['is_host_rj'] = creating_host_location(dataframe)

    count_characters_variables(dataframe, config.string_variables)
    return dataframe
