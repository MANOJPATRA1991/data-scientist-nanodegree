import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self):
        '''
        I didn't have any required attributes needed when creating my class.
        '''


    def fit(self, reviews_pth, movies_pth, latent_features=12, learning_rate=0.0001, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        '''
        # Store inputs as attributes
        self.reviews = pd.read_csv(reviews_pth)
        self.movies = pd.read_csv(movies_pth)

        # Create user-item matrix
        user_item = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        # Users as rows and movies as columns
        self.user_item_df = user_item.groupby(['user_id','movie_id'])['rating'].max().unstack()
        self.user_item_matrix = np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful values to be used through the rest of the function
        # Number of users
        self.n_users = self.user_item_matrix.shape[0]
        # Number of movies
        self.n_movies = self.user_item_matrix.shape[1]
        # Number of non-null ratings
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_matrix))
        # List of user ids 
        self.user_ids_series = np.array(self.user_item_df.index)
        # List of movie ids
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # Initialize the user and movie matrices with random values
        user_feature_matrix = np.random.rand(self.n_users, self.latent_features)
        movie_feature_matrix = np.random.rand(self.latent_features, self.n_movies)

        # Initialize sse (Sum of squared errors) at 0 for first iteration
        sse_accum = 0

        # Keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # For each iteration
        for iter in range(self.iters):

            # Update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # If the rating exists
                    if self.user_item_matrix[i, j] > 0:

                        # Compute the error as the difference of the actual and the dot product of the user and movie latent features
                        diff = self.user_item_matrix[i, j] - np.dot(user_feature_matrix[i, :], movie_feature_matrix[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # Update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_feature_matrix[i, k] += self.learning_rate * (2 * diff * movie_feature_matrix[k, j])
                            movie_feature_matrix[k, j] += self.learning_rate * (2 * diff * user_feature_matrix[i, k])

            # Print results
            print("%d \t\t %f" % (iter + 1, sse_accum / self.num_ratings))

        # SVD based fit
        # Keep user_feature_matrix and movie_feature_matrix for safe keeping
        self.user_feature_matrix = user_feature_matrix
        self.movie_feature_matrix = movie_feature_matrix

        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, user_id, movie_id):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        try:
            # Get user placeholder index
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            
            # Get movie placeholder index
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_feature_matrix[user_row, :], self.movie_feature_matrix[:, movie_col])

            # Get the movie name
            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie'])[5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None


    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                        given movie, or recs for a user_id given
        '''
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            # If the user is available from the matrix factorization data,
            # use FunkSVD and rank movies based on the predicted values
            # For use with user indexing
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # Take the dot product of that row and the V matrix
                preds = np.dot(self.user_feature_matrix[idx,:],self.movie_feature_matrix)

                # Pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # If we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names


    def create_train_test(self, order_by, training_size, testing_size):
        '''    
        INPUT:
        reviews - (pandas df) dataframe to split into train and test
        order_by - (string) column name to sort by
        training_size - (int) number of rows in training set
        testing_size - (int) number of columns in the test set
        
        OUTPUT:
        training_df -  (pandas df) dataframe of the training set
        validation_df - (pandas df) dataframe of the test set
        '''
        reviews_new = self.reviews.sort_values(order_by)
        training_df = reviews_new.head(training_size)
        validation_df = reviews_new.iloc[training_size:training_size+testing_size]
        
        return training_df, validation_df

    def validation_comparison(self, val_df):
        '''
        INPUT:
        val_df - the validation dataset created in the third cell above
            
        OUTPUT:
        rmse - RMSE of how far off each value is from it's predicted value
        perc_rated - percent of predictions out of all possible that could be rated
        actual_v_pred - a 10 x 10 grid with counts for actual vs predicted values
        preds - (list) predictions for any user-movie pairs where it was possible to make a prediction
        acts - (list) actual values for any user-movie pairs where it was possible to make a prediction
        '''
        val_users = np.array(val_df['user_id'])
        val_movies = np.array(val_df['movie_id'])
        val_ratings = np.array(val_df['rating'])
        
        sse = 0
        num_rated = 0
        preds, acts = [], []
        actual_v_pred = np.zeros((10, 10))
        
        for idx in range(len(val_users)):
            try:
                pred = self.predict_rating(val_users[idx], val_movies[idx])
                sse += (pred - val_ratings[idx]) ** 2
                num_rated += 1
                preds.append(pred)
                acts.append(val_ratings[idx])
                actual_v_pred[11 - int(val_ratings[idx] - 1), int(round(pred) - 1)] += 1
            except:
                continue
                
        rmse = np.sqrt(sse / num_rated)
        perc_rated = num_rated / len(val_users)
        
        
        return rmse, perc_rated, actual_v_pred, preds, acts

if __name__ == '__main__':
    import recommender as r

    # Instantiate recommender
    rec = r.Recommender()

    # Fit recommender
    rec.fit(reviews_pth='data/train_data.csv', movies_pth= 'data/movies_clean.csv', learning_rate=.01, iters=1)

    # Predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # User in the dataset
    print(rec.make_recommendations(1,'user')) # User not in dataset
    print(rec.make_recommendations(1853728)) # Movie in the dataset
    print(rec.make_recommendations(1)) # Movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)

    # Use our function to create training and test datasets for reviews
    train_df, val_df = rec.create_train_test('date', 8000, 2000)

    # Create user-by-item matrix - this will keep track of order of users and movies in u and v
    train_user_item = train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
    train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
    train_data_np = np.array(train_data_df)

    rmse, perc_rated, actual_v_pred, preds, acts = rec.validation_comparison(val_df)
    print(rmse, perc_rated, actual_v_pred, preds, acts)

