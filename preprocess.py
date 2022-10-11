import pandas as pd


def get_processed_dataframes(investment_relationship_path: str,
                             company_details_path: str):
    """
        Reads and does some basic pre-processing of the investment_relationship
        and company details tables provided by crunchbase and returns the processed
        tables along with a table which has information just about the investments

    :param investment_relationship_path: path to csv file
        which has investment relationship on your local machine

    :param company_details_path: path to csv file which
        has company details on your local machine

    :return: investment_relationship, investments, invested_companies_detail
    """
    # Read investment_relationship table
    investment_relationship = pd.read_csv(investment_relationship_path, dtype=str)

    # money_invested_usd is null
    investment_relationship.drop([
        "Unnamed: 0", "id",
        "money_invested_usd",
        "crunchbase_investment_id"
    ], axis=1, inplace=True)

    id_cols = ["crunchbase_company_id",
               "crunchbase_person_id",
               "invested_in_company_id",
               "funding_round_id"]

    # Typecasting
    investment_relationship = investment_relationship.astype({
        'money_raised_usd': 'float',
        'is_lead_investor': 'bool'
    })
    investment_relationship[id_cols] = investment_relationship[id_cols].astype(float).fillna(0).astype(int)

    # Trimming the date string
    investment_relationship["announced_on"] = investment_relationship["announced_on"].str[:10]

    investment_relationship.fillna(0, inplace=True)

    # Read company details table
    company_details = pd.read_csv(company_details_path, dtype=str)

    # Typecasting
    count_cols = ["total_funding_rounds_count", "total_investments_count"]
    company_details[count_cols] = company_details[count_cols].astype(float).astype(int)
    company_details = company_details.astype({
        'id': 'int',
        'total_funding_usd': 'float'
    })

    # Filling empty values of description first with short
    # description then with name and then with empty string
    company_details['description'] = (
        company_details.description
        .fillna(company_details.short_description)
        .fillna(company_details.name)
        .fillna('')
    )

    # Trimming the date string
    company_details["founded_on"] = company_details['founded_on'].str[:10]

    company_details.drop([
        "Unnamed: 0",
        "company_id",
        "permalink",
        "name",
        "short_description"
    ], axis=1, inplace=True)

    # Let's get just the investments in the companies removing the investor information
    investments = (
        investment_relationship
        .drop([
            'crunchbase_company_id',
            'crunchbase_person_id',
            'is_lead_investor'
        ], axis=1)
        .drop_duplicates()
        .rename(columns={'invested_in_company_id': 'id'})
    )

    # Now let's get details of companies which have been invested in
    invested_companies_detail = company_details[company_details.id.isin(investments.id.unique())]

    # Find proxy foundation date as the first investment date where foundation dates isn't available
    proxy = investment_relationship.groupby('invested_in_company_id').announced_on.min().reset_index()
    proxy.columns = ['id', 'announced_on']

    invested_companies_detail = invested_companies_detail.merge(proxy, on='id', how='left')

    # Filling empty founded_on with first investment date
    invested_companies_detail['founded_on'] = (
        invested_companies_detail.founded_on.fillna(invested_companies_detail.announced_on)
    )
    invested_companies_detail.drop('announced_on', axis=1, inplace=True)

    return investment_relationship, investments, invested_companies_detail
