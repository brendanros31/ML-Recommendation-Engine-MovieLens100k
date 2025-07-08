from surprise.model_selection import cross_validate
from surprise import SVD


svd = SVD()


def cv(df, measures, num_validations, verbose=True):
    _cv = cross_validate(svd, 
                        df, 
                        measures=measures, 
                        cv=num_validations, 
                        verbose=verbose)
    return _cv 



# Generating recomendations - Unseen movies
def recomendations(df_movies, ratings_suprise, num_recs, userID):
    # Create user_1 - copy df_movies
    user = df_movies.copy()
    user = user.reset_index()

    # Training SVD model on full dataset
    trainset = ratings_suprise.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    # Generating Predictions for User 1 in column
    user['Estimate_Score'] = user['movieId'].apply(lambda x: svd.predict(userID,x).est)

    # Cleaning predictions
    user = user.drop(['movieId','index'], axis=1)

    #Display recomendations
    return user.head(num_recs)