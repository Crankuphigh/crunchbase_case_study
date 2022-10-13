import pandas as pd
from utils import diff_in_years
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


def create_investment_score(investment_relationship: pd.DataFrame,
                            grouped_investments: pd.DataFrame,
                            company_details: pd.DataFrame):
    """

    :param investment_relationship:
    :param grouped_investments:
    :param company_details:
    :return:
    """

    # Compute rank of each alphabetical series of investment found in data for a company
    grouped_investments['ranks'] = grouped_investments.groupby(['id', 'series']).cumcount() + 1

    scores = []

    # Let's define score for each investment found in the investment relationship table
    for index, row in investment_relationship.iterrows():
        overall_score_invested_company = (company_details[company_details.id == row.invested_in_company_id]
        ['performance_in_group'].iloc[0])
        subdf = grouped_investments[(grouped_investments.id == row.invested_in_company_id)]
        rank = subdf[subdf.series == row.series].ranks.values()
        if rank + 1 in subdf.ranks:
            # factor by which next investment has grown
            invest_growth_ratio = (subdf[subdf.ranks == rank + 1].money_raised_usd.iloc[0] /
                                   subdf[subdf.ranks == rank].money_raised_usd.iloc[0])

            # years in which next investment is raised
            invest_duration_diff = diff_in_years(subdf[subdf.ranks == rank + 1].announced_on.iloc[0],
                                                 subdf[subdf.ranks == rank].announced_on.iloc[0])

            # average of growth of investment ratio, inverse of duration in which it has been raised
            # and overall company performance penalised by the rank of round in which it has been seen
            investment_score = (invest_growth_ratio
                                + 1 / invest_duration_diff
                                + overall_score_invested_company / rank) / 3
        else:
            investment_score = overall_score_invested_company / rank
        scores.append(investment_score)

    investment_relationship['scores'] = scores
    return investment_relationship
