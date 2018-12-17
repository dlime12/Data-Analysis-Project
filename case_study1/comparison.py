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
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

# Import ROC AUC library
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



# Print new line
def NewLine():
	print("\n")


def Features(model, features, index=20):
	importances = model.feature_importances_
	
	# Descending order
	indices = np.argsort(importances)
	indices = np.flip(indices, axis=0)
	
	indices = indices[:index]
	
	for i in indices:
		print(features[i], ':', importances[i])


def ROC():

	""" Decision Tree """
	# Import Pre-processed data into dataframe (df)
	df = data_prep()
	# Random State number to keep the results consistent
	rs = 10

	# y axis, and x axis
	y = df['G3']
	x = df.drop(['G3'], axis=1)

	x_matrix = x.values
	x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, stratify=y, random_state=rs)


	params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 7), 'min_samples_leaf': range(20, 60, 10)}

	cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
	cv.fit(x_train, y_train)

	dt_model = cv.best_estimator_
	print(dt_model)

	# Acc score
	y_pred_dt = dt_model.predict(x_test)
	dt_acc = accuracy_score(y_test, y_pred_dt)


	# ROC Index
	y_pred_proba_dt = dt_model.predict_proba(x_test)
	roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])


	# ROC graph
	fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])






	""" Logistic Regression """
	df = data_prep()
	# list columns to be transformed
	columns_to_transform = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']
	# transform the columns with np.log
	for col in columns_to_transform:
		df[col] = df[col].apply(lambda x: x + 1)
		df[col] = df[col].apply(np.log)

	y = df['G3']

	X = df.drop(['G3'], axis=1)
	# setting random state
	rs = 10
	X_mat = X.as_matrix()
	X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

	# initialise a standard scaler object
	scaler = StandardScaler()
	# learn the mean and std.dev of variables from training data
	# then use the learned values to transform training data
	X_train = scaler.fit_transform(X_train, y_train)
	# use the statistic that you learned from training to transform test data
	X_test = scaler.transform(X_test)

	params = {'C': [pow(10, x) for x in range(-6, 4)]}
	# use all cores to tune logistic regression with C parameter
	cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train, y_train)
	log_reg_model = cv.best_estimator_
	#print(log_reg_model)


	log_reg_model = cv.best_estimator_
	print(log_reg_model)

	# Acc score
	y_pred_log_reg = log_reg_model.predict(X_test)
	log_reg_acc = accuracy_score(y_test, y_pred_log_reg)


	# ROC index
	y_pred_proba_log_reg = log_reg_model.predict_proba(X_test)
	roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])


	# ROC graph
	fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])








	""" Neural Networks """
	# Import Pre-processed data into dataframe (df)
	df = data_prep()
	# Random State number to keep the results consistent
	rs = 10


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

	# transform log
	X_train_rfe = rfe.transform(X_train_log)
	X_test_rfe = rfe.transform(X_test_log)



	# Train the model
	params = {'hidden_layer_sizes': [(2,)], 'alpha': [0.01]}

	cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(max_iter=800, random_state=rs), cv=10, n_jobs=-1)
	cv.fit(X_train_rfe, y_train_log)

	nn_model = cv.best_estimator_
	print(nn_model)

	# NN acc score
	y_pred_nn = cv.predict(X_test_rfe)
	nn_acc = accuracy_score(y_test_log, y_pred_nn)


	# ROC index
	y_pred_proba_nn = nn_model.predict_proba(X_test_rfe)
	roc_index_nn = roc_auc_score(y_test_log, y_pred_proba_nn[:, 1])


	# ROC graph
	fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test_log, y_pred_proba_nn[:,1])






	""" Accuracy Scores """
	print("\n")
	print("Accuracy score on test for DT:", dt_acc)
	print("Accuracy score on test for logistic regression:", log_reg_acc)
	print("Accuracy score on test for NN:", nn_acc)









	""" ROC Index Scores """
	print("\n")
	print("ROC index on test for DT:", roc_index_dt)
	print("ROC index on test for logistic regression:", roc_index_log_reg)
	print("ROC index on test for NN:", roc_index_nn)





	""" ROC Curve Graph """
	plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
	plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
	plt.plot(fpr_nn, tpr_nn, label='ROC Curve for NN {:.3f}'.format(roc_index_nn), color='darkorange', lw=0.5)
	

	# plt.plot(fpr[2], tpr[2], color='darkorange',
	# lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()









if __name__ == '__main__':



	#print("\n\n")
	#compare_dt()
	#compare_log_reg()
	#compare_nn()
	ROC()