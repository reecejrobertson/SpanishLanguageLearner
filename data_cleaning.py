#data_cleaning.py
import pandas as pd
import numpy as np

def clean_data(file='Vol3ProjectData.csv'):
    '''
    Function to clean the data from an input file.
    This is the script version of Jonathan's cleaning Jupyter Notebook, modified
    to be able to call within a notebook

    Returns
        Tuple (DF, DF):
            df: DataFrame object of cleaned data
            users_who_star: df but limited on only the users who have starred words
    '''
    df = pd.read_csv(file,sep='\t')

    # display(df)

    #Change all the values in the prioritized column to strings.
    df['prioritized'] = df['prioritized'].astype(str)
    df = df[df['prioritized'] != "\\N"]

    #Change all the timestamps to ints.
    df = df[df['updated_timestamp'] != "\\N"]
    df['updated_timestamp'] = df['updated_timestamp'].astype(np.int64)

    #Change all the user IDs to strings.
    df = df[df['user_id'] != "\\N"]
    df['user_id'] = df['user_id'].astype(str)

    #Change all the concept IDs to strings.
    df = df[df['concept_id'] != "\\N"]
    df['concept_id'] = df['concept_id'].astype(str)

    # Add the starred and mistaken rows.
    # Starred words have priority 2 or 3; mistaken words have priority 1 or 3.
    df['starred'] = df['prioritized'].astype(np.int64) > 1
    df['mistaken'] = df['prioritized'].astype(np.int64) % 2 == 1

    #reset the indices
    df = df.reset_index(drop = True)

    # display(df)

    #Create the words_studied column:

    # Cast the user IDs to ints for faster sorting.
    df['user_id'] = df['user_id'].astype(np.int64)

    #get all the unique values in user_id
    unique = df['user_id'].unique()
    #create a list of zeros the same length of the dataframe
    words_studied = np.zeros(len(df['user_id']))
    #cylce through each unique user_id (this takes about 5 min to run)
    for i in unique:
        #define a dataframe where the user IDs match the value we are checking in the unique IDs
        sf = df.loc[df['user_id'] == i]
        #sort that dataframe by the timestamps
        sf = sf.sort_values('updated_timestamp')
        #gather those indices
        sf = sf.index
        #cycle through those indices and add the number of word that it is that they have learned at the index in words_studied
        for i in range(1,len(sf)+1):
            words_studied[sf[i-1]] = i
    #create the column and add the list to that column
    df['words_studied'] = words_studied.astype(np.int32)

    # Cast the user IDs back to strings.
    df['user_id'] = df['user_id'].astype(str)

    # display(df)

    # Get the set of user_ids for users who have starred at least one word.
    starred = df.loc[df['starred'] == True]
    user_ids_who_star = set(starred['user_id'])

    # Get all the rows of each user who has starred a word.
    users_who_star = df.loc[df['user_id'].isin(user_ids_who_star)]
    # display(users_who_star)

    return df, users_who_star

# # Save the cleaned data to a csv.
# df.to_csv('Vol3CleanedData.csv')
# users_who_star.to_csv('Vol3StarredData.csv')
