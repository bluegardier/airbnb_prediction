import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    :param price: Price value with str type.
    :return: Price as int type.
    """
    price = int(price[1:-3].replace(",", ""))
    return price
