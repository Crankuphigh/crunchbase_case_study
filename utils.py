import pandas as pd
from datetime import timedelta


def diff_in_years(start_date, end_date):
    """
        Calculates difference between dates in years.
        e.g.  diff_in_years('2020-01-01', '2021-03-01') = 1.16

    :param start_date: start date

    :param end_date: end date

    :return: age in years (float)
    """
    diff_in_days = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    return diff_in_days / timedelta(days=365)
