import helpers as helper
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import pickle


class StarBucksBase:
    def __init__(self, porfolio_df, profile_df, transcript_df, preprocessing_required=True):
        self.succ_tried_offers = None
        self.model = None
        self.gcv = None
        # Create mapping of portfolio offer id to custom type name
        self.portfolio_offers = {}
        
        self.portfolio = porfolio_df
        self.profile = profile_df
        self.transcript = transcript_df
        
        if preprocessing_required:
            self.__data_preprocessing()


    def __data_preprocessing(self):
        self.__preprocess_portfolio()
        self.__preprocess_profile()
        self.__preprocess_transcript()


    def __preprocess_portfolio(self):
        """
        Preprocess the portfolio dataframe
        """

        # Rename id column
        self.portfolio.rename(columns = {'id':'offer_id'}, inplace = True)

        # Create a mapping of offer id to its details
        for i in range(self.portfolio.shape[0]):
            row = self.portfolio.iloc[i]
            self.portfolio_offers[row['offer_id']] = row['offer_type'].upper() + '/C' + str(row['difficulty']) + '/R' + str(row['reward']) + '/T' + str(row['duration'])

        # Update offer_id values to be values in the offer_id dictionary
        self.portfolio['offer_id'] = self.portfolio['offer_id'].apply(lambda x: self.portfolio_offers[x])


    def __preprocess_profile(self):
        """
        Preprocess the profile dataframe
        """
        # Convert to datetime format
        self.profile['became_member_on'] = pd.to_datetime(self.profile['became_member_on'], format='%Y%m%d')

        # Create new column to add start year of membership
        self.profile['member_year'] = self.profile['became_member_on'].dt.year

        # Calculate number of days since the membership began
        self.profile['member_since_days'] = np.abs(self.profile['became_member_on'] - datetime.now()).dt.days.apply(self.__create_member_group)

        # Create age groups - 1,2,3,4
        self.profile['age_group'] = self.profile['age'].apply(self.__create_age_group)

        # Rename the id column index
        self.profile.rename(columns = {'id':'person'}, inplace = True)

        # Drop became_member_on column as it is no longer required
        self.profile.drop('became_member_on', axis=1, inplace=True)

        # Replace age 118 with NaN as it is a placeholder for null values
        self.profile['age'] = self.profile['age'].apply(lambda x: np.nan if x == 118 else x)

        # Get profiles without null values
        self.profile = self.profile[self.profile['age'].isnull() == False]

        # Create income group based on income ranges
        self.profile['income_group'] = self.profile['income'].apply(self.__create_income_group)


    def __preprocess_transcript(self):
        """
        Preprocess the transcript dataframe
        """
        # Create type of value: offer_id or amount
        self.transcript['type'] = self.transcript['value'].apply(lambda x : 'offer_id' if (list(x.keys())[0] == 'offer id' or list(x.keys())[0] == 'offer_id') else list(x.keys())[0])

        # Create column value id containing offer id or type_0
        self.transcript['offer_id'] = self.transcript['value'].apply(lambda x : self.portfolio_offers[list(x.values())[0]] if (list(x.keys())[0] == 'offer id' or list(x.keys())[0] == 'offer_id') else 'NoPromotionApplied')

        # Filter out transcripts with type of offer_id and merge with the profile dataframe
        # as we are only interested in the transcripts with offers
        transcript_offer = self.transcript[self.transcript['type'] == 'offer_id']
        transcript_offer = pd.merge(transcript_offer, self.profile, how='left', on='person')
        transcript_offer = transcript_offer.dropna()

        # Merge it with the portfolio dataframe
        transcript_offer = pd.merge(transcript_offer, self.portfolio, how='left', on='offer_id')

        # Filter out records with offer received events
        transcript_offer_received = transcript_offer[transcript_offer['event'] == 'offer received']
        transcript_offer_received.rename(index=str, columns={"time":"time_received"}, inplace=True)
        transcript_offer_received.drop(['event'], axis=1, inplace=True)

        # Filter out records with offer viewed events
        transcript_offer_viewed = transcript_offer[transcript_offer['event'] == 'offer viewed']
        transcript_offer_viewed.rename(index=str, columns={"time": "time_viewed"}, inplace=True)
        cols_to_drop = ['event', 'value', 'type', 'age', 'gender', 'income', \
                        'member_year', 'difficulty', 'duration', 'reward', 'channels', \
                        'member_since_days', 'offer_type', 'age_group', 'income_group']
        transcript_offer_viewed.drop(cols_to_drop, axis=1, inplace=True)

        # Filter out records with offer completed events
        transcript_offer_completed = transcript_offer[transcript_offer['event']=='offer completed']
        transcript_offer_completed.rename(index=str, columns={"time":"time_completed"}, inplace=True)
        transcript_offer_completed.drop(cols_to_drop, axis=1, inplace=True)

        # Merge the three dataframes created above
        all_offers = pd.merge(transcript_offer_received, transcript_offer_viewed, on=['person', 'offer_id'], how='left')
        all_offers = pd.merge(all_offers, transcript_offer_completed, on=['person', 'offer_id'], how='left')

        # Get valid data based on time window
        all_offers = all_offers[((all_offers['time_viewed'] >= all_offers['time_received']) &\
                                ((all_offers['time_completed'] >= all_offers['time_viewed'])
                                | all_offers['time_completed'].isnull())) |
                                (all_offers['time_viewed'].isnull() &\
                                all_offers['time_completed'].isnull())]

        # Calculate the time of expiry for each offer type
        time_of_expiry_df = pd.DataFrame({ 'time_of_expiry': np.array(all_offers['time_received']) + np.array(all_offers['duration']) })
        all_offers = pd.concat([all_offers, time_of_expiry_df], axis=1)

        # An offer is considered successful if time of receipt is less than time of view, time of view is less than time of completion and time of completion is less than time of expiry [PAID]
        all_offers['successful_offer'] = ((all_offers['time_received'] <= all_offers['time_viewed']) & (all_offers['time_viewed'] <= all_offers['time_completed']) & (all_offers['time_completed'] <= all_offers['time_of_expiry'])).apply(lambda x: int(x == True))

        # An offer is considered tried if time of receipt is less than time of view, time of view is less than time of expiry and time of completion is greater than time of expiry [PURCHASED AFTER PROMO EXPIRED] or NEVER PURCHASED
        all_offers['tried_offer'] = ((all_offers['time_received'] <= all_offers['time_viewed']) & (all_offers['time_viewed'] <= all_offers['time_of_expiry']) & ((all_offers['time_of_expiry'] < all_offers['time_completed']) | (np.isnan(all_offers['time_completed'])))).apply(lambda x: int(x == True))

        # An offer is considered a failure if it is neither successful nor tried
        all_offers['failed_offer'] = ((all_offers['successful_offer'] == 1) | (all_offers['tried_offer'] == 1)).apply(lambda x: int(x == False))

        # Remove duplicate records that might have been created as a result of the merge operations
        all_offers.drop_duplicates(subset=['time_received', 'person', 'offer_id'], keep='first', inplace=True)

        # Drop the row with all null values
        all_offers = all_offers.drop(4)

        # Get the transaction amount details along with time spent to do the transaction
        transcript_transaction = self.transcript[self.transcript['type']=='amount']
        transcript_transaction['amount_spent'] = transcript_transaction['value'].apply(lambda x: list(x.values())[0])
        transcript_transaction.drop(['event', 'value', 'type', 'offer_id'], axis=1, inplace=True)
        transcript_transaction.rename(index=str, columns={"time": "time_spent"}, inplace=True)

        # Get non-failure offers => success or tried
        self.succ_tried_offers = all_offers[(all_offers['successful_offer']==1) | (all_offers['tried_offer']==1)]

        # Merge with transactions as we are interested in the transaction details of non-failure offers
        self.succ_tried_offers = pd.merge(self.succ_tried_offers, transcript_transaction, on='person', how='left')

        # Create column indicating transaction done during the offer period or after the offer period
        self.succ_tried_offers['spent_during_offer'] = self.succ_tried_offers.apply(self.__check_offer_transactions, axis=1)
    
    
    def __create_member_group(self, days):
        """
        Create member group based on membership days till date

        INPUT:
            days: Number of days since beginning of the membership

        OUTPUT:
            String or NaN
        """
        if days == 744:
            return 'since_2_year'
        elif 744 <= days <= 3*365:
            return 'since_3_year'
        elif 3*365 <= days <= 4*365:
            return 'since_3_year'
        elif 4*365 <= days <= 5*365:
            return 'since_4_year'
        elif 5*365 <= days <= 6*365:
            return 'since_5_year'
        elif 6*365 <= days <= 7*365:
            return 'since_6_year'
        elif 7*365 <= days <= 8*365:
            return 'since_7_year'
        else:
            return np.nan


    def __create_age_group(self, age):
        """
        Create age group

        INPUT:
            days: Age of the individual

        OUTPUT:
            Integer or NaN
        """
        if age <= 38:
            return 1
        elif 39 <= age <= 54:
            return 2
        elif 55 <= age <=73:
            return 3
        elif 74 <= age <= 101:
            return 4
        else:
            return np.nan


    def __create_income_group(self, income):
        """
        Create age group

        INPUT:
            days: Age of the individual

        OUTPUT:
            Integer or NaN
        """
        if income < 45000:
            return 1
        elif 45000 <= income < 60000:
            return 2
        elif 60000 <= income < 75000:
            return 3
        elif 75000 <= income < 90000:
            return 4
        elif 90000 <= income < 105000:
            return 5
        elif 105000 <= income < 120000:
            return 6
        else:
            return np.nan


    def __check_offer_transactions(self, df):
        """
        Check if transaction is done during the offer period or after the offer period

        INPUT: 
            df: Transcript dataframe

        OUTPUT:
            1 or 0 indicating the truthness of the condition mentioned
        """
        if df['successful_offer'] == 1:
            return 1 if (df['time_spent'] >= df['time_received']) and (df['time_spent'] <= df['time_completed']) else 0
        else:
            return 1 if (df['time_spent'] >= df['time_received']) and (df['time_spent'] <= df['time_of_expiry']) else 0


    def encode_categorical_variables(self, df):
        # Encode channels as new one-hot encoded columns
        df = helper.encode_column_with_list(df, col='channels', prefix='channel_')

        # Encode the offer type column
        df = helper.create_dummy_df(df, ['offer_type'], False, custom_prefix='offer')

        # Create dummy variables for gender, income and age group values
        df = helper.create_dummy_df(df, ['gender', 'income_group', 'age_group'], False)

        # Create dummy variables for member_since_days values
        df = helper.create_dummy_df(df, ['member_since_days'], False, custom_prefix='member')

        return df


    def create_test_and_train(self, test_size=0.2):
        """
        Create train and test sets
        
        INPUT:
            test_size: Train test split ratio
        
        OUTPUT:
            X_train, X_test, y_train, y_test
        """
        transaction_cols = [
            'age', 'income', 'member_year', 'member_since_days', 'gender', \
            'reward', 'difficulty', 'duration', 'channels', 'offer_type'
        ]
        X = self.succ_tried_offers[transaction_cols]
        X = self.encode_categorical_variables(X)

        y = self.succ_tried_offers['amount_spent']
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        return X_train, X_test, y_train, y_test
    
    
    def train_model(self, X_train, y_train, save_path=None, model_type='svr'):
        if (model_type == 'svr'):
            self.model = SVR(kernel='poly', degree=7)
            self.model.fit(X_train, y_train)
            print('SVR model trained on 7-degree poly kernel')
        elif (model_type == 'rf'):
            self.model = RandomForestRegressor(n_estimators=100, max_depth=20, max_features=15, min_samples_split=5)
            self.model.fit(X_train, y_train)
            print('RF Model trained with 100 estimators')
        else:
            params = {'n_estimators' : [10, 50, 100], 'max_depth' : [5, 10, 30, 80], \
                        'max_features': [1, 3, 8, 15], 'min_samples_split': [3, 5, 10, 30, 50, 100]}
            g_rfm = RandomForestRegressor(random_state=1024)
            self.gcv = GridSearchCV(g_rfm, params, verbose=10, cv=5, scoring='r2')
            self.gcv.fit(X_train, y_train)
            print("Best parameters: ", self.gcv.best_params_)
            self.model = RandomForestRegressor(max_depth=self.gcv.best_params_['max_depth'], max_features=self.gcv.best_params_['max_features'], min_samples_split=self.gcv.best_params_['min_samples_split'], n_estimators=self.gcv.best_params_['n_estimators'])
            self.model.fit(X_train, y_train)

        if save_path is not None:
            self.save_model(save_path)
        
        
    def save_model(self, filename):
        """
        Save the model

        INPUT: 
            filename: File to save to
        """
        pickle.dump(self.model, open(filename, 'wb'))
        

    def read_model(self, filename):
        """
        Read the saved model

        INPUT: 
            filename: File to read from
        """
        # Open a file, where you stored the pickled data
        file = open(filename, 'rb')

        # Dump information to that file
        self.model = pickle.load(file)

        # Close the file
        file.close()
        
    def model_predict(self, X_test, y_test, model_type):
        y_preds = self.model.predict(X_test)
        print('{} Model R2 score: '.format(model_type), r2_score(y_test, y_preds))