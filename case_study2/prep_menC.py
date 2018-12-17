import pandas as pd
import numpy as np

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Scaling the values so it suits for clustering
from sklearn.preprocessing import StandardScaler

# K-Means clustering
from sklearn.cluster import KMeans




# Data Preparation for Clustering (Task 1)


# Not skipping empty values, to demonstrate data pre-processing steps
df = pd.read_csv('dataset/MENS_CLOTHING_SALES_2018.csv', na_filter=False)

# Pre-process dataset ready for cleaning (Question 1, Unusual data-types)
def pre_process_dataset():
	global df # Use the local df variable

	# Replaces the empty values as NaN (Not a Number) in 'AnnualSales' column 
	df['AnnualSales'] = df['AnnualSales'].replace('', np.nan).astype(float)

	# Replaces the empty values as NaN (Not a Number) in 'Sales' column 
	df['Sales'] = df['Sales'].replace('', np.nan).astype(float)

	# Replaces the empty values as NaN (Not a Number) in 'TotalInvestment' column 
	df['TotalInvestment'] = df['TotalInvestment'].replace('', np.nan).astype(float)

	# One-hot encoding
	df = pd.get_dummies(df)




# Visualise each columns to find any hidden problems
def visualise_data_column(column_name):
	# Don't include the StoreCode (a.k.a unique ID of stores) 
	if column_name != 'StoreCode':
		print('\n', df[column_name].describe()) # Describes the column
		print('\n', df[column_name].value_counts(bins=10)) # Prints in gorups of 10
		print('\n', df[column_name].unique()) # Prints unique values
		column_vis_dist = sns.distplot(df[column_name].dropna()) # Plots the column for visual representation
		plt.show()

# Plot the columns
def visualise_each_column():
	global df # Use the local df variable  
	for column in df:
			visualise_data_column(column)






# Drop unused variables
def finalised_dataset():
	global df # Use the local df variable
	for column in df:
		if column == 'StoreCode':
			df = df.drop(['StoreCode'], axis=1)

		# Remove the Sales column from the analysis
		elif column == "Sales":
			df = df.drop(['Sales'], axis=1)
		
		else:
			continue # Ignore

		"""
		# un-Comment for testing (and comment out sales column)
		# Remove the Sales column from the analysis
		if column == 'AnnualSales':
			df = df.drop(['AnnualSales'], axis=1)

		# Remove the AnuualSales column from the analysis
		if column == 'SFloorSize':
			df = df.drop(['SFloorSize'], axis=1)
		"""




# Display the process of dataframe cleaning
def show_dataframe_clean_process():
	global df # Use the local df variable

	print("Non pre-processed data \n")
	print("\n", df.head(50)) # Display the dataframe rows
	pre_process_dataset() # Pre Process data, ready for cleaning
	df.info() # Display the attributes of each variable

	# Visualise data
	print("\n")
	visualise_each_column()

	# Eliminate errorneous values
	print("\n", "Rows # before dropping errorneous rows", len(df), "\\400") # Before elimination
	
	df = df[df['SFloorSize'] > 1] # Drop the errorneous rows
	
	print("Row # after dropping errorneous rows", len(df), "\\400") # After elimination
	visualise_each_column() # Re-visualise, confirming the elimination

	# Finalise the dataset, (drop unused variables ['StoreCode'])
	finalised_dataset()
	print("\n Finalised data \n")
	df.info()
	print("\n", df.head(50)) # Display the dataframe rows

	return df

# Show the process
# show_dataframe_clean_process() # Un-comment here to see the full process



# Identify which store is underperforming in sales
def underperforming_store():
	global df # Use the local df variable
	pre_process_dataset() # Pre Process data ready for cleaning
	df = df[df['SFloorSize'] > 1] # Drop the errorneous rows
	df = df.drop(['AnnualSales', 'SFloorSize', 'TotalInvestment'], axis=1)
	visualise_each_column()
	df = df.sort_values('Sales')
	pd.set_option('display.max_rows', None)
	print("\n", df.head(385))
	df = df[df['Sales'] != 300] # Drop the under-performing store 
	return df

# Identify underperforming_stores
# underperforming_stores() # Un-comment here to see the underperforming store


# Return the dataframe used for clustering
def get_dataframe_MENSC():
	global df # Use the local df variable
	pre_process_dataset() # Pre Process data ready for cleaning
	df = df[df['SFloorSize'] > 1] # Drop the errorneous rows
	df = df[df['Sales'] != 300] # Drop the under-performing store (task 3)
	finalised_dataset() # Finalise dataset (drop unused variables)
	return df
# Call this function to retrieve the dataframe or for testing