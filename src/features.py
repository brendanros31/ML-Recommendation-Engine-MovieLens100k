import pandas as pd

from surprise import Reader, Dataset



# Converting the dataset into Surprise's format
def format_surpsise(df, columns):
    reader = Reader()   # Initialize the Reader object
    return Dataset.load_from_df(df[columns], reader)



# Finding User's most 4 star rated movies
def preferences(df_movies, df_ratings, num_prefs, userID):
    # Movies rated 4 stars by user_1
    ratings = df_ratings[(df_ratings['userId']==userID) & (df_ratings['rating']==4)]

    # Set movieId as the index - easier joining
    ratings = ratings.set_index('movieId')

    # Combine ratings_1, df_movies - to get movie titles
    ratings = ratings.join(df_movies)['title']

    # Top 10 movies rated 4 stars by user_1
    return pd.DataFrame(ratings.head(num_prefs))


