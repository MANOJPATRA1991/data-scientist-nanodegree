import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the data
    
    Args:
        messages_filepath: File path to read messages data
        categories_filepath: File path to read categories data
    
    Returns:
        A pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="outer", on=["id"])
    return df


def clean_data(df):
    """
    Cleans the data
    
    Args:
        df: A pandas DataFrame
    
    Returns:
        A pandas DataFrame
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = [category_colname.split('-')[0] for category_colname in row.values]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    categories['related'] = categories['related'].map(lambda x: 1 if x == 2 else x)
    categories.drop('child_alone', axis=1, inplace=True)
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], join="inner", axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the data to the database
    
    Args:
        df: A pandas DataFrame
        database_filename: File to store the data at
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists="replace")  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()