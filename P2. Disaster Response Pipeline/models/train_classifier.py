import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the starting verb of a sentence and
    creates a new feature for the classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Load data from the database
    
    Args:
        database_filepath: File to read the data from
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponseTable", engine)
    return df


def tokenize(text):
    """
    Tokenize the text
    
    Arguments:
        text: A text message
    Output:
        A list of tokens generated from the input
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    Build the model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', AdaBoostClassifier())
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [0.01, 0.1],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on the test set
    
    Args:
        model: Classifier model to be evaluated
        X_test: Test example feature set
        Y_test: Test labels
        category_names: List of category names
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the model for future use
    
    Args:
        model: Model to be saved
        model_filepath: File location at which to save the model
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()