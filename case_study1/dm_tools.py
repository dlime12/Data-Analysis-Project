# dm_tools.py
import numpy as np
import pandas as pd

from io import StringIO
from sklearn.tree import export_graphviz

def data_prep():
	df = pd.read_csv('STUDENT.csv')


	# Change the values in schoolsup column to binary
	df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})

	# Change the values in famsup column to binary
	df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})

	# Change the values in paid column to binary
	df['paid'] = df['paid'].map({'no': 0, 'yes': 1})

	# Change the values in activities column to binary
	df['activities'] = df['activities'].map({'no': 0, 'yes': 1})

	# Change the values in higher column to binary
	df['higher'] = df['higher'].map({'no': 0, 'yes': 1})

	# Change the values in internet column to binary
	df['internet'] = df['internet'].map({'no': 0, 'yes': 1})

	# Change the values in G3 column to binary
	df['G3'] = df['G3'].map({'FAIL': 0, 'PASS': 1})


	# Drop unused variables
	df.drop(['id', 'InitialName', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus','Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery', 'romantic', 'G1', 'G2'], axis=1, inplace=True) 


	# one-hot encoding
	df = pd.get_dummies(df)

	return df
