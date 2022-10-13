from __future__ import print_function
import argparse
import sys

from preprocess import get_processed_dataframes
from feature_engineering import create_company_features, create_investment_score


def main():
    parser = argparse.ArgumentParser(description='Get crunchbase case study results')
    parser.add_argument('-ir_path', '--investment_relationship_path', required=True,
                        help='investment relationship table path')
    parser.add_argument('-cd_path', '--company_details_path', required=True, help='company details table path')
    parser.add_argument('-c_id', '--crunchbase_company_id', required=True, help='crunchbase company id')
    parser.add_argument('-p_id', '--crunchbase_person_id', required=True, help='crunchbase person id')

    args = parser.parse_args(sys.argv[1:])

    investment_relationship, grouped_investments, invested_companies_detail = get_processed_dataframes(
        args.ir_path, args.cd_path
    )
    invested_companies_detail = create_company_features(invested_companies_detail)
    investment_relationship = create_investment_score(investment_relationship,
                                                      grouped_investments,
                                                      invested_companies_detail)

    if args.crunchbase_company_id:
        investor_score = investment_relationship[
            investment_relationship.crunchbase_company_id == args.crunchbase_company_id
            ].scores.mean()
    elif args.crunchbase_person_id:
        investor_score = investment_relationship[
            investment_relationship.crunchbase_person_id == args.crunchbase_person_id
            ].scores.mean()
    else:
        raise Exception("Pass at least one of the following: crunchbase_company_id, crunchbase_person_id")

    print("Score for the investor is: ", investor_score)


if __name__ == '__main__':
    main()
