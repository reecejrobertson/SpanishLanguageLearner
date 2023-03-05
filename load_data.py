#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix 

def encode(df, expanded=True, column_labels=[]):
    if len(column_labels)==0:
        column_labels = df['concept_id'].unique()
    row_labels = df['user_id'].unique()
    #Initialize Matrix
    # Layer 0 is a T/F (0/1) indicator if the word has been learned at all
    # Layer 1 is a T/F (0/1) indicator if the word has been starred or mistaken
    df = df.sort_values(by=['user_id','words_studied'])
    unique = list(set(df['user_id']))
    unique = np.array(unique)
#     print(unique)
#     print(df['user_id'].values[0])
#     print(np.where(unique == df['user_id'].values[0])[0][0])
    #if we want to expand the matrix (connor needed the data from the unexpanded matrix sometimes)
    if expanded:
        #create an empty 3 dimensional matrix full of zeros
#         data = np.zeros((len(row_labels),len(column_labels),len(column_labels)))
        data = csr_matrix((len(df)-len(row_labels),len(column_labels)), dtype=np.int8)
        labels = np.zeros(len(df)-len(row_labels))

        n = 0
        #cycle through every user and add ones to the words theyve learned in order 
        #(data[i][0] will have a 1 at the first word user i learned after 1 iteration, data[i][1] will have a 1 at the first and second word user i learned after 2 iterations)
        for i in range(len(row_labels)):
            for j in range(list(df['user_id'].values).count(df['user_id'].values[n])-1):
                if j > 0:
                    data[i,j] = data[i,j-1] 
                data[i,j,np.where(column_labels == df.values[n][1])[0][0]] = 1
                labels[i] = np.where(column_labels == df.values[n+1][1])[0][0]
                n += 1
        return column_labels, data, labels
    #if not expanded then just do all the ones at once and create 2 matrices (one for encounters of words and one for starred/mistaken words)                
    else:
        data = np.zeros((len(row_labels),len(column_labels),2))
        for index, row in df.iterrows():
            i = np.where(row_labels == row['user_id'])[0][0]
            j = np.where(column_labels == row['concept_id'])[0][0]
            #Add indices and data to lists to construct matrix
            data[i,j,0] = 1
            data[i,j,1] = max((row['starred'], row['mistaken']))
        return column_labels, data


def train_test_split(test_size = 5, file='Vol3StarredData.csv', X=None):
    """
    Loads in training data from file and prepares user test data
    
    Parameters:
        test_size (float): proportion of provided data to make into test cases
        file (str): name of file from which to load data
    Returns:
        X_train (DataFrame): Data from all users not included in test sets
                            will have (1-test_size) proportion of the data
        tups (list): list of tuples containing (X_test, y_test) data pairs
    """
    if X is None:
        # Read in data
        X = pd.read_csv(file)
        #Set new index column after converting to int
        X['Unnamed: 0'] = X['Unnamed: 0'].astype('int')
        X = X.set_index('Unnamed: 0')
    # get all the unique users
    users = np.unique(X['user_id'])
    #Choose n random users
    test_users = np.random.choice(users, test_size)
    # Create test and training set
    tups = []
    X_train = X.copy()
    for i in test_users:
        #Exclude user data from training data
        X_train = X_train[X_train["user_id"] != i]
        #Extract User data and sort by word progress
        user_data = X[X["user_id"] == i].sort_values(by=['words_studied'])
        #Get newest 50% of words and try to predict those
        #Oldest 50% of words are used as the inputs
        X_test = user_data.iloc[:int(len(user_data)*.5)]
        y_test = user_data.iloc[int(len(user_data)*.5):]
        tups.append((X_test, y_test))
    
    return X_train, tups
        

