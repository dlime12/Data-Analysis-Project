import pandas as pd
from apyori import apriori
import seaborn as sns
import matplotlib.pyplot as plt
# load the transaction dataset
df = pd.read_csv('POS_TRANSACTIONS_2018 .csv')
# info and the first 10 transactions
# print(df.info())
# print(df.head(10))
def prepare(df):
    # group by Transaction_Id, then list all services
    transactions = df.groupby(['Transaction_Id'])['Product_Name'].apply(list)
    #print(transactions.head(20))

    # type cast the transactions from pandas into normal list format and run apriori
    transaction_list = list(transactions)
    results = list(apriori(transaction_list, min_support=0.02))
    return results


def convert_apriori_results_to_pandas_df(results):
    rules = []
    for rule_set in results:
        for rule in rule_set.ordered_statistics:
            # items_base = left side of rules, items_add = right side
            # support, confidence and lift for respective rules
            rules.append([','.join(rule.items_base), ','.join(
                rule.items_add), rule_set.support, rule.confidence, rule.lift])
    # typecast it to pandas df
    return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift'])

def association():
    results = prepare(df)
    result_df = convert_apriori_results_to_pandas_df(results)
    # sort all acquired rules descending by lift
    print("Sorted by Lift")
    result_df = result_df.sort_values(by='Lift', ascending=False)
    print(result_df.head(20))

    # sort all acquired rules descending by Confidence
    print("Sorted by Confidence")
    result_df = result_df.sort_values(by='Confidence', ascending=False)
    print(result_df.head(20))
    plotRules(result_df.head(20));

# plot rules
def plotRules(result_df):
    sup = sns.barplot(x=result_df.index, y="Support", data=result_df)
    plt.show();
    conf = sns.barplot(x=result_df.index, y="Confidence", data=result_df)
    plt.show();
    lift = sns.barplot(x=result_df.index, y="Lift", data=result_df)
    plt.show();

# Exercise book subset
def exBook():
    results = prepare(df)
    result_df = convert_apriori_results_to_pandas_df(results)
    left_df = result_df[result_df['Left_side']=="Exercise book"]
    right_df = result_df[result_df['Right_side']=="Exercise book"]
    result_df = left_df
    result_df = result_df.append(right_df)
    print("Number of rules:")
    print(result_df.shape[0])
    print("Sorted by Lift")
    result_df = result_df.sort_values(by='Lift', ascending=False)
    print(result_df)
    print("Sorted by Confidence")
    result_df = result_df.sort_values(by='Confidence', ascending=False)
    print(result_df)
