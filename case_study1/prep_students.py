import pandas as pd
import numpy as np


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
    df.drop(['id', 'InitialName', 'school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
        'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'nursery', 'romantic', 'G1', 'G2'], axis=1, inplace=True) 


    # one-hot encoding
    df = pd.get_dummies(df)
   
    return df



def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_

    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)
    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]
    for i in indices:
        print(feature_names[i], ':', importances[i])
 
 
def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file
