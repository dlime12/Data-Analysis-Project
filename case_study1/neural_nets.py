# libraries
import pandas as pd
import numpy as np

# Data Processing function for this specific dataset
from prep_students import *

# Neural Network libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# Parameter tuner
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Multilayer perceptron classifier
from sklearn.neural_network import MLPClassifier

# Import RFECV and Logistic Regression for Recursive Feature Elimination
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

# Import Decision Tree for Recursive Feature Elimination
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# Import ROC AUC library
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# Neural Networks model built on default parameters (Question #1)
def Default_Model():
	# First default neural networks model
	model = MLPClassifier(random_state=rs)
	model.fit(X_train, y_train)

	print("Train accuracy:", model.score(X_train, y_train))
	print("Test accuracy:", model.score(X_test, y_test))

	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))

	print(model)


# GridSearchCV processes
def Optimise_GridSearchCV():
	# Print number of rows and columns used in the training data set
	# As the hidden_layer_sizes should be no more than its input variable 
	# And also, no less than output nodes (1)
	print(X_train.shape, "\n")

	# Print number of rows and columns used in the training data set (and use it to range the hidden_layer_sizes)
	params = {'hidden_layer_sizes': [(x,) for x in range(1, 17, 1)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

	# Find optimal hidden layer size & alpha parameter values with GridSearchCV
	# Set up the model with adjusted parameters, set the max_iter to 1000 just in case (200 isn't enough)
	cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=1000, random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train, y_train)

	# Print the Train / Test accuracy
	print("Train accuracy:", cv.score(X_train, y_train))
	print("Test accuracy:", cv.score(X_test, y_test))

	# Classification report
	y_pred = cv.predict(X_test)
	print(classification_report(y_test, y_pred))

	# Print the best hidden_layer_sizes value
	print(cv.best_params_)



# Neural Networks model built with GridSearchCV for optimal performance (Question #2)
def GridSearchCV_Refined_Model():

	# Remake the MLPClassifier with the optimal hyperparameter values
	model = MLPClassifier(alpha=0.01, hidden_layer_sizes=(1,), max_iter=300, random_state=rs)
	model.fit(X_train, y_train)

	# Print its Train & Test accuracy
	print("Train accuracy:", model.score(X_train, y_train))
	print("Test accuracy:", model.score(X_test, y_test))

	# Print classification report
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))

	# Print its model information
	print(model, '\n')




# Log transformation process for improvement of performance
def Dimensionality_Reduction():
	# list columns to be transformed
	columns_to_transform = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']

	df_log = df.copy()

	# transform the columns with np.log
	for col in columns_to_transform:
		df_log[col] = df_log[col].apply(lambda x: x + 1)
		df_log[col] = df_log[col].apply(np.log)


	# create X, y and train test data partitions
	y_log = df_log['G3']
	X_log = df_log.drop(['G3'], axis=1)
	X_mat_log = X_log.as_matrix()
	X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, random_state=rs)

	# standardise them again
	scaler_log = StandardScaler()
	X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
	X_test_log = scaler_log.transform(X_test_log)



	# After transforming the variable, train and tuen model, to see any improvements
	params = {'hidden_layer_sizes': [(x,) for x in range(1, 17, 1)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

	# Build model
	cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=1000, random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train_log, y_train_log)

	# Print accuracy
	print("Train accuracy:", cv.score(X_train_log, y_train_log))
	print("Test accuracy:", cv.score(X_test_log, y_test_log))

	# Classification report
	y_pred = cv.predict(X_test_log)
	print(classification_report(y_test_log, y_pred))
	
	# Print best parameters
	print(cv.best_params_)

	



def RFE_Log():
	# list columns to be transformed
	columns_to_transform = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']

	df_log = df.copy()

	# transform the columns with np.log
	for col in columns_to_transform:
		df_log[col] = df_log[col].apply(lambda x: x + 1)
		df_log[col] = df_log[col].apply(np.log)


	# create X, y and train test data partitions
	y_log = df_log['G3']
	X_log = df_log.drop(['G3'], axis=1)
	X_mat_log = X_log.as_matrix()
	X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, random_state=rs)

	# standardise them again
	scaler_log = StandardScaler()
	X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
	X_test_log = scaler_log.transform(X_test_log)


	rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
	rfe.fit(X_train_log, y_train_log)

	print(rfe.n_features_)




	# transform log
	X_train_rfe = rfe.transform(X_train_log)
	X_test_rfe = rfe.transform(X_test_log)


	# After transforming the variable, train and tuen model, to see any improvements
	params = {'activation': ['relu', 'identity', 'tanh', 'logistic'], 'hidden_layer_sizes': [(x,) for x in range(1, 17, 1)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}


	cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=800, random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train_rfe, y_train_log)
	
	print("Train accuracy:", cv.score(X_train_rfe, y_train_log))
	print("Test accuracy:", cv.score(X_test_rfe, y_test_log))
	
	y_pred = cv.predict(X_test_rfe)
	print(classification_report(y_test_log, y_pred))
	
	print(cv.best_params_)






def RFE_Dec_Tree():
	# list columns to be transformed
	columns_to_transform = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']

	df_log = df.copy()

	# transform the columns with np.log
	for col in columns_to_transform:
		df_log[col] = df_log[col].apply(lambda x: x + 1)
		df_log[col] = df_log[col].apply(np.log)


	# create X, y and train test data partitions
	y_log = df_log['G3']
	X_log = df_log.drop(['G3'], axis=1)
	X_mat_log = X_log.as_matrix()
	X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, random_state=rs)

	# standardise them again
	scaler_log = StandardScaler()
	X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
	X_test_log = scaler_log.transform(X_test_log)


	params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 7), 'min_samples_leaf': range(20, 60, 10)}

	cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
	cv.fit(X_train_log, y_train_log)

	print(cv, "\n")

	analyse_feature_importance(cv.best_estimator_, X_log.columns)


	selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
	X_train_sel_model = selectmodel.transform(X_train)
	X_test_sel_model = selectmodel.transform(X_test)
	print("\n", X_train_sel_model.shape)



	# After transforming the variable, train and tuen model, to see any improvements
	params = {'hidden_layer_sizes': [(x,) for x in range(1, 17, 1)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}


	cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=800, random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train_sel_model, y_train)

	print("Train accuracy:", cv.score(X_train_sel_model, y_train))
	print("Test accuracy:", cv.score(X_test_sel_model, y_test))

	y_pred = cv.predict(X_test_sel_model)
	print(classification_report(y_test, y_pred))

	print(cv.best_params_)


	






if __name__ == '__main__':
	# Import Pre-processed data into dataframe (df)
	df = data_prep()

	# Random State number to keep the results consistent
	rs = 10

	# train test split
	y = df['G3']
	X = df.drop(['G3'], axis=1)
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train, y_train)
	X_test = scaler.transform(X_test)




	#Optimise_GridSearchCV()
	#print("\n\n")
	#Default_Model()
	#print("\n\n")
	#GridSearchCV_Refined_Model()
	#print("\n\n")
	#Dimensionality_Reduction()
	#print("\n\n")
	#RFE_Log()
	#print("\n\n")
	#RFE_Dec_Tree()
	#print("\n\n")
	#Compare_Models_AllVars()
	#print("\n\n")
	#Compare_Models_RFEVars()
	#print("\n\n")
	#compare()
