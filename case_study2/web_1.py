import pandas as pd
import datetime
from collections import defaultdict

# ip,date_time,request,step,session,user_id
names=['Host', 'Datetime', 'Request', 'Step', 'Session', 'User']
df = None

# Removes the given file type (i.e .js requests from the frame)
def removeUndesirable(type):
	global df

	# Filter undesirable
	mask = (df['Request'].str.endswith(type))
	# print("# Rows before:", len(df))

	# Invert
	df2 = df[~mask]
	# print("After images removal", len(df2)) 

	# Set
	df = df2

# Setup options and dataframe
def initiate():
	global df

	pd.set_option('display.max_rows', 1000)
	pd.set_option('display.max_columns', 1000)
	pd.set_option('display.width', 1000)

	df = pd.read_csv('WEB_LOG_DATA.csv', sep=',', names=names, header=None) 
	df.drop(0, inplace=True)

# Correcting and issues in the data and correctly setting datetime
def errorCorrection():
	global df

	# correct the incorrect dataframe types
	df['User'] = df['User'].astype(int)
	df['Session'] = df['Session'].astype(int)
	df['Step'] = df['Step'].astype(int)
	df['Date time'] = pd.to_datetime(df['Datetime'], format='%d/%b/%Y:%H:%M:%S', errors='coerce') # 03/Sep/2017:09:00:50          [18/Apr/2005:21:25:07 +1000]
	df = df.drop(['Datetime'], axis=1)

# Return the dataframe
def getDF():
	global df

	initiate()
	errorCorrection()
	removeUndesirable('.js')
	removeUndesirable('.ico')

	# Replace trailing forward slashes
	df['Request'] = df['Request'].str.replace(r'\/$', '')

	return df
