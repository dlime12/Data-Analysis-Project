# @author Jack Teys
# 
# For this file to work, pydot-ng, pydot, and graphviz need to be either installed or upgraded.
# Also uncomment one of the imports below using the correct Python ver
# 

import numpy as np
import pandas as pd
import pydot
import os
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from prep_students import *

# Python 3
from io import StringIO

# Python 2.7
#from io import BytesIO as StringIO

# This is a hack to get Graphviz to work (instead of adding Graphviz to the system PATH)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Variables to drop and random state
drop = ['id', 'InitialName', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery', 'romantic', 'G1', 'G2']
rs = 10

global x, x_train, x_test
global y, y_train, y_test

# Print new line
def NewLine():
	print("\n")

# Pre decision tree operations
def PreOps():
	# Get globals for setting values
	global x, x_train, x_test
	global y, y_train, y_test
	


	# y axis, and x axis
	y = df['G3']
	x = df.drop(['G3'], axis=1)

	x_matrix = x.values
	x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, stratify=y, random_state=rs)


def Features(model, features, index=20):
	importances = model.feature_importances_
	
	# Descending order
	indices = np.argsort(importances)
	indices = np.flip(indices, axis=0)
	
	indices = indices[:index]
	
	for i in indices:
		print(features[i], ':', importances[i])

def PrintTree(model, columns, name):
	dotfile = StringIO()
	export_graphviz(model, out_file=dotfile, feature_names=columns)
	graph = pydot.graph_from_dot_data(dotfile.getvalue())

	# Export tree to text file (this way it includes label names)
	f = open(name + '.txt', 'w')
	f.write(dotfile.getvalue())

	graph[0].write_png(name + '.png')

def DecisionTree():
	model = DecisionTreeClassifier(random_state=rs)
	model.fit(x_train, y_train)

	# print(model)
	
	# Training and test scores from the model
	training_score = model.score(x_train, y_train)
	test_score = model.score(x_test, y_test)
	
	# Print the scores
	print("--Default DecisionTree--")
	print("X and Y training score: " + str(training_score))
	print("X and Y test score: " + str(test_score))

	y_pred = model.predict(x_test)
	print(classification_report(y_test, y_pred))

	Features(model, x.columns)


# GridSearchCV #1
def GridSearch():
	params = {'criterion': ['gini', 'entropy'], 'max_depth': range(2, 7), 'min_samples_leaf': range(20, 60, 10)}

	cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
	cv.fit(x_train, y_train)

	# Accuracy scores
	NewLine()
	print("--GridSearchCV--")
	print("Train accuracy:", cv.score(x_train, y_train))
	print("Test accuracy:", cv.score(x_test, y_test))

	# Classification report
	y_pred = cv.predict(x_test)
	print(classification_report(y_test, y_pred))

	# Best performing paramaters and feature list
	print(cv.best_params_)
	NewLine()
	Features(cv.best_estimator_, x.columns)


	ROCChart(cv.best_estimator_)

def ROCChart(model):
	# Model prediction
	y_pred = model.predict(x_test)
	
	# Probability
	y_pred_proba_dt = model.predict_proba(x_test)
	
	print("Probability produced by decision tree for each class vs actual prediction on G3 (0 = fail, 1 = pass).")
	print("(Probs on zero)\t(probs on one)\t(prediction made)")
	
	# Top 10
	for i in range(20):
		print(y_pred_proba_dt[i][0], '\t', y_pred_proba_dt[i][1], '\t', y_pred[i])
	
	y_pred_proba_dt = model.predict_proba(x_test)
	roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])
	
	print("ROC index on test for DT:", roc_index_dt)

	fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])

	plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
	plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


df = data_prep()
PreOps()
DecisionTree()
GridSearch()
