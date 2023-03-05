df = users_who_star.copy()

column_labels = df['concept_id'].unique()
row_labels = df['user_id'].unique()
#Initialize Matrix
# Layer 0 is a T/F (0/1) indicator if the word has been learned at all
# Layer 1 is a T/F (0/1) indicator if the word has been starred or mistaken
data = np.zeros((len(row_labels),len(column_labels),2))
for index, row in df.iterrows():
    i = np.where(row_labels == row['user_id'])[0][0]
    j = np.where(column_labels == row['concept_id'])[0][0]
    #Add indices and data to lists to construct matrix
    data[i,j,0] = 1
    data[i,j,1] = max((row['starred'], row['mistaken']))
