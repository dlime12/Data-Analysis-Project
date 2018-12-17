import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
import seaborn as sns
import matplotlib.pyplot as plt


def data_prep():

    df = pd.read_csv('STUDENT.csv')

    # Drop unused variables
    df.drop(['id', 'InitialName', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
             'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery', 'romantic', 'G1', 'G2'], axis=1, inplace=True)

    # Fill the missing values for G1 coulumn
    # NaN is replaced by the mean value rounded to 1 decimal point
    #df['G1'].fillna(round(df['G1'].mean(), 1), inplace=True)

    # Fill the missing values for G2 coulumn
    # NaN is replaced by the mean value rounded to 1 decimal point
    #df['G2'].fillna(round(df['G2'].mean(), 1), inplace=True)

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

    # one-hot encoding
    df = pd.get_dummies(df)

    return df

# Task 3 Question 1 and Question 2


def regression():
    df = data_prep()

    # plot columns to determine if any require transformation ( uncomment the following line )
    # plotCol(df)

    # list columns to be transformed
    columns_to_transform = ['traveltime', 'famrel', 'freetime',
                            'goout', 'Walc', 'Dalc', 'health', 'absences']
    # transform the columns with np.log
    for col in columns_to_transform:
        df[col] = df[col].apply(lambda x: x + 1)
        df[col] = df[col].apply(np.log)
    # plot them again to show the distribution ( uncomment the following line )
    # plotCol(df)

    y = df['G3']

    X = df.drop(['G3'], axis=1)
    # setting random state
    rs = 10
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    # initialise a standard scaler object
    scaler = StandardScaler()
    # learn the mean and std.dev of variables from training data
    # then use the learned values to transform training data
    X_train = scaler.fit_transform(X_train, y_train)
    # use the statistic that you learned from training to transform test data
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=rs)
    # fit it to training data
    model.fit(X_train, y_train)

    # training and test accuracy
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))
    # classification report on test data
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # get variables and their coefficient to the model
    print("Variables used:")
    feature_names = X.columns
    coef = model.coef_[0]
    # sort them out in descending order
    indices = np.argsort(np.absolute(coef))
    indices = np.flip(indices, axis=0)
    for i in indices:
        print(feature_names[i], ':', coef[i])

    # grid search CV
    print("")
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    # use all cores to tune logistic regression with C parameter
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(
        random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    # test the best model
    print("Train accuracy:", cv.score(X_train, y_train))
    print("Test accuracy:", cv.score(X_test, y_test))
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))
    # print parameters of the best model
    print(cv.best_params_)

# Task 3 Question 3
def rfeRegression():
    df = data_prep()
    # list columns to be transformed
    columns_to_transform = ['traveltime', 'studytime', 'failures',
                            'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']
    # transform the columns with np.log
    for col in columns_to_transform:
        df[col] = df[col].apply(lambda x: x + 1)
        df[col] = df[col].apply(np.log)

    y = df['G3']

    X = df.drop(['G3'], axis=1)
    # setting random state
    rs = 10
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    # initialise a standard scaler object
    scaler = StandardScaler()
    # learn the mean and std.dev of variables from training data
    # then use the learned values to transform training data
    X_train = scaler.fit_transform(X_train, y_train)
    # use the statistic that you learned from training to transform test data
    X_test = scaler.transform(X_test)

    # Regression with RFE
    rfe = RFECV(estimator=LogisticRegression(random_state=rs), cv=10)
    rfe.fit(X_train, y_train)

    X_train_sel = rfe.transform(X_train)
    X_test_sel = rfe.transform(X_test)

    # grid search CV
    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    # use all cores to tune logistic regression with C parameter
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(
        random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train_sel, y_train)
    # test the best model
    print("Train accuracy:", cv.score(X_train_sel, y_train))
    print("Test accuracy:", cv.score(X_test_sel, y_test))
    y_pred = cv.predict(X_test_sel)
    print(classification_report(y_test, y_pred))
    # print parameters of the best model
    print(cv.best_params_)

    model = LogisticRegression(random_state=rs)
    # fit it to training data
    model.fit(X_train_sel, y_train)

    print("")
    # comparing how many variables before and after
    print("Original feature set", X_train.shape[1])
    print("Number of features after elimination", rfe.n_features_)
    # get variables and their coefficient to the model
    print("Variables used:")
    feature_names = X.columns
    coef = model.coef_[0]
    # sort them out in descending order
    indices = np.argsort(np.absolute(coef))
    indices = np.flip(indices, axis=0)
    for i in indices:
        print(feature_names[i], ':', coef[i])

def compare():
    df = data_prep()
    # list columns to be transformed
    columns_to_transform = ['traveltime', 'studytime', 'failures',
                            'famrel', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']
    # transform the columns with np.log
    for col in columns_to_transform:
        df[col] = df[col].apply(lambda x: x + 1)
        df[col] = df[col].apply(np.log)

    y = df['G3']

    X = df.drop(['G3'], axis=1)
    # setting random state
    rs = 10
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    # initialise a standard scaler object
    scaler = StandardScaler()
    # learn the mean and std.dev of variables from training data
    # then use the learned values to transform training data
    X_train = scaler.fit_transform(X_train, y_train)
    # use the statistic that you learned from training to transform test data
    X_test = scaler.transform(X_test)

    params = {'C': [pow(10, x) for x in range(-6, 4)]}
    # use all cores to tune logistic regression with C parameter
    cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(
        random_state=rs), cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    log_reg_model = cv.best_estimator_
    print(log_reg_model)

    # Accuracy
    y_pred_log_reg = log_reg_model.predict(X_test)
    print("Accuracy score on test for logistic regression:", accuracy_score(y_test, y_pred_log_reg))

    # Classification report
    print("classification report:")
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))

    # The Area under an ROC curve (ROC AUC)
    y_pred_proba_log_reg = log_reg_model.predict_proba(X_test)
    roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:,1])
    print("ROC index on test for logistic regression:", roc_index_log_reg)
    fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])
    plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
    plt.show()


# the function is used to determine which of the variables might need transformation
def plotCol(df):
    # setting up subplots for easier visualisation
    f, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=False)

    sns.distplot(df['traveltime'].dropna(), hist=False, ax=axes[0, 0])
    sns.distplot(df['studytime'].dropna(), hist=False, ax=axes[0, 1])
    sns.distplot(df['failures'].dropna(), hist=False, ax=axes[0, 2])
    sns.distplot(df['schoolsup'].dropna(), hist=False, ax=axes[0, 3])
    sns.distplot(df['famsup'].dropna(), hist=False, ax=axes[1, 0])
    sns.distplot(df['paid'].dropna(), hist=False, ax=axes[1, 1])
    sns.distplot(df['activities'].dropna(), hist=False, ax=axes[1, 2])
    sns.distplot(df['higher'].dropna(), hist=False, ax=axes[1, 3])
    sns.distplot(df['internet'].dropna(), hist=False, ax=axes[2, 0])
    sns.distplot(df['famrel'].dropna(), hist=False, ax=axes[2, 1])
    sns.distplot(df['freetime'].dropna(), hist=False, ax=axes[2, 2])
    sns.distplot(df['goout'].dropna(), hist=False, ax=axes[2, 3])
    sns.distplot(df['Walc'].dropna(), hist=False, ax=axes[3, 0])
    sns.distplot(df['Dalc'].dropna(), hist=False, ax=axes[3, 1])
    sns.distplot(df['health'].dropna(), hist=False, ax=axes[3, 2])
    sns.distplot(df['absences'].dropna(), hist=False, ax=axes[3, 3])

    plt.show()
