from apyori import apriori
from task6 import *

def convert_apriori_results_to_pandas_df(results):
	rules = []
	
	for rule_set in results:
		for rule in rule_set.ordered_statistics:
			# items_base = left side of rules, items_add = right side
			# support, confidence and lift for respective rules
			rules.append([','.join(rule.items_base), ','.join(rule.items_add), rule_set.support, rule.confidence, rule.lift])
	
	# Cast to Pandas dataframe
	return pd.DataFrame(rules, columns=['Left_side', 'Right_side', 'Support', 'Confidence', 'Lift'])

# Sort list by type. Print sorted values by count.
def printOrderBys(type, count, order=False):
	global result_df
	results = result_df.sort_values(by=type, ascending=order)
	print(type + " values")
	print(results.head(count))

# Display entire columns
pd.set_option('display.max_colwidth', -1)

df = getDF() # From task6.py
result_df = None

# names=['Host', 'Datetime', 'Request', 'Step', 'Session', 'User']
transactions = df.groupby(['Session'])['Request'].apply(list)

# Cast to Python list
transaction_list = list(transactions)
results = list(apriori(transaction_list, min_support=0.02))

result_df = convert_apriori_results_to_pandas_df(results)

# Order by lift
printOrderBys('Lift', 100)

# Order by confidence
printOrderBys('Confidence', 100)