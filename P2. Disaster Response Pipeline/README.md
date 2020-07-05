# Disaster Response Pipeline

## Project Setup

1. Create a virtual environment with the command `python -m venv env`.

2. Activate the virtual environment with `source env/bin/activate` on **MacOS** or with `.\env\Scripts\activate` on **Windows**.

3. Run `pip install -r requirements.txt` to install dependencies.

## Program Execution

1. **Run ETL pipeline**: The ETL pipeline loads the data, cleans the data and stores it in a database. To run the ETL pipeline, use `python data/process_data.py`

2. **Run ML Pipeline**: The ML pipeline trains the classifier, searches for best hyperparameters and saves the trained model. To run the ML pipeline, use `python models/train_classifier.py`

3. **Run the app**: To run the app, use `python app/run.py`, and then navigate to the URL mentioned in the terminal `http://0.0.0.0:<port>/`. The port will vary. 

## Data Visualization

### Distribution of messages by genre

![Mesages by genre](https://github.com/MANOJPATRA1991/data-scientist-nanodegree/blob/master/P2.%20Disaster%20Response%20Pipeline/Distribution%20of%20message%20genres.jpg)

### Distribution of messages by category

![Messages by category](https://github.com/MANOJPATRA1991/data-scientist-nanodegree/blob/master/P2.%20Disaster%20Response%20Pipeline/Distribution%20of%20message%20categories.jpg)

## Conclusion

We tried many classifiers based on the conditions such as the dataset contains less than 50000 instances and the dataset is imbalanced.
Classifiers that we tried:
1. AdaBoost
2. Random forest
3. Naive Bayes

Due to the imbalanced dataset, not much improvement is seen across these classifiers.

Some of the approaches to solve this problem are:
1. Random Undersampling and Oversampling
2. Class weights in the models
3. Change your Evaluation Metric
4. Collect more Data
5. Treat the problem as anomaly detection: Use autoencoders for anomaly detection
6. Using boosting models
