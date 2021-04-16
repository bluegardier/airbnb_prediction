import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from airbnb_prediction import objects


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

    df['regiao'] = np.where(df['neighbourhood_cleansed'].isin(objects.centro), 'centro',
                            np.where(df['neighbourhood_cleansed'].isin(objects.zona_sul), 'zona_sul',
                                     np.where(df['neighbourhood_cleansed'].isin(objects.zona_norte), 'zona_norte',
                                              np.where(df['neighbourhood_cleansed'].isin(objects.zona_norte),
                                                       'zona_norte',
                                                       np.where(df['neighbourhood_cleansed'].isin(objects.zona_oeste),
                                                                'zona_oeste', 'not_found')
                                                       )
                                              )
                                     )
                            )

    return df['regiao']
