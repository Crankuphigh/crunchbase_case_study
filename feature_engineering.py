import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def create_company_features(company_details: pd.DataFrame):
    """
        Computes company specific features like, age of a company,
        group in which the company belongs (based on different groups present in data)
        performance metric of a company, performance in the specific group for that company

    :param company_details: Dataframe consisting of columns:
        'description',
        'founded_on',
        'total_funding_usd',
        'total_funding_rounds_count'

    :return: Dataframe consisting the features
    """

    # This is age in years with min age as 1 years
    company_details['age'] = 2023 - company_details.founded_on.str[:4].astype(int)

    # Funding per million per round per year of existence
    company_details['performance'] = (
            company_details.total_funding_usd /
            (company_details.total_funding_rounds_count * company_details.age * 1000000)
    )

    # Group Similar Companies
    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(list(company_details.description.values))
    company_details['group'] = KMeans(n_clusters=20, init='k-means++', max_iter=200, n_init=10).fit(X).labels_

    # Average performance seen in any group
    avg_perf = company_details.groupby('group').performance.mean().reset_index()
    avg_perf.columns = ['group', 'avg_perf']
    company_details = company_details.merge(avg_perf, on='group', how='left')

    # Performance ratio with mean performance in group
    company_details['performance_in_group'] = company_details.performance / company_details.avg_perf
    company_details.drop('avg_perf', axis=1, inplace=True)

    return company_details


def create_investment_features(investments: pd.DataFrame):
    """
        Computes rank of each alphabetical series of investment found in data for a company

    :param investments: Dataframe consisting of columns:
        'id',
        'series'
    :return: Dataframe consisting the ranks
    """

    investments['series_rank_in_data'] = investments.groupby(['id', 'series']).cumcount() + 1
    return investments
