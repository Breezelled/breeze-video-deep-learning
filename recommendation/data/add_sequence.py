import pandas as pd

df = pd.read_csv('query_recommendation_dataset.csv', header=None,
                 names=["user_id", "username", "gender", "interest_type", "sentiment_rating", "info_id", "director",
                        "writer", "star", "language", "company", "country", "type", "liked", "favorited"])

# group by user_id and liked/favorited, combine info_id to a string, separate by '|'
liked_seq = df.loc[df['liked'] == 't'].groupby('user_id')['info_id'].apply(lambda x: '|'.join(x)).reset_index(
    name='liked_sequence')
favorited_seq = df.loc[df['favorited'] == 't'].groupby('user_id')['info_id'].apply(lambda x: '|'.join(x)).reset_index(
    name='favorited_sequence')

# add 2 new cols
df = pd.merge(df, liked_seq, on='user_id', how='left')
df = pd.merge(df, favorited_seq, on='user_id', how='left')

df.to_csv('dataset_header.csv', index=False)
