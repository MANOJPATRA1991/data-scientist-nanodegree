# Starbuck's Capstone Challenge

## Dataset overview
The data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.

Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. 

People produce various events, including receiving offers, opening offers, and making purchases.

As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. 

1. BOGO offer: a user needs to spend a certain amount to get a reward equal to that threshold amount.
2. Discount: a user gains a reward equal to a fraction of the amount spent. 
3. Informational offer: there is no reward, but neither is there a requisite amount that the user is expected to spend. 

Offers can be delivered via multiple channels: Email, Mobile, Social, Web

## Project Motivation

**GOAL**:  To use the data to 

a) Identify which groups of people are most responsive to each type of offer.

b) How best to present each type of offer?

c) How many people across different categories actually completed the transaction in the offer window?

d) Which individual attributes contributed the most during the offer window?

## Data Dictionary

1. `profile.json`: Rewards program users (17000 users x 5 fields)
    | Name | Type | Description |
    | - | - | - |
    | gender | categorical | M, F, O, or null |
    | age | numeric | missing value encoded as 118 |
    | id | string/hash |
    | became_member_on | date | format YYYYMMDD |
    | income | numeric | individual income |
2. `portfolio.json`: Offers sent during 30-day test period (10 offers x 6 fields)
    | Name | Type | Description |
    | - | - | - |
    | reward | numeric | money awarded for the amount spent |
    |channels | list | web, email, mobile, social |
    |difficulty | numeric | money required to be spent to receive reward |
    | duration | numeric | time for offer to be open, in days |
    | offer_type | string | bogo, discount, informational |
    | id | string/hash |

3. `transcript.json`: Event log (306648 events x 4 fields)
    | Name | Type | Description |
    | - | - | - |
    | person | string/hash |
    | event | string | offer received, offer viewed, transaction, offer completed |
    | value | dictionary | different values depending on event type |
    | offer id | string/hash | not associated with any "transaction" |
    | amount | numeric | money spent in "transaction" |
    | reward | numeric | money gained from "offer completed" |
    | time | numeric | hours after start of test |

## Project Structure

```
|
| - data: Stores original and preprocessed data for easy usage
|   |
|   | - portfolio.json
|   |
|   | - profile.json
|   |
|   | - transcript.json
|   |
|   | - preprocessed_portfolio.csv
|   |
|   | - preprocessed_profile.csv
|   |
|   | - preprocessed_transcripts.csv
|   |
|   | - preprocessed_succ_tried_offers.csv
|
| - models: Stores the trained models
|   |
|   | - finalized_model.sav
|   |
|   | - rf_model.sav
|   |
|   | - svr_model.sav
|
| - helpers.py: Helper functions reside here
|
| - starbucks.py: The StarbucksBase class is the heart of this project, it includes methods related to preprocessing,   merging dataframes, creating test and train sets, model training, evaluation, prediction and saving.
|
| - README.md
|
| - structured.ipynb: This notebook contains code related to preprocessing data and evaluation post project restructuring
```

## Conclusion

### Q1. Identify which groups of people are most responsive to each type of offer.

Members who joined one to two years back relative to 2018, seem to be equally responsive to each offer while the trend decreases with new members (1 year or less than 1 year) and old members (more than two years).

Males in age group of 1 (< 38 years) and 3 (between 55 years and 73 years) dominated across almost all the offers while females showed domination in group 3 (between 55 years and 73 years).

Males in income groups 2 (45000 <= income < 60000) and 3 (60000 <= income < 75000) showed domination for informational discounts.

### Q2. How best to present each type of offer?

| Offer type | Web | Mobile | Email | Social | OFFER_TOTAL |
| - | - | - | - | - | - |
| Bogo | 7385 | 10779 | 10779 | 9541 | 38484 |
| Discount | 8311 | 7366 | 8311 | 6024 | 30012 |
| Informational | 2131 | 6187 | 6187 | 4056 | 18561 |
| **TOTAL** | 17827 | 24332 | 25227 | 19621 |

### Q3. How many people across different categories actually completed the transaction in the offer window?

Approximately 12.8% of the total population for offer type BOGO completed the transaction in the offer window; 12% for offer type discount and 18% for informational type.

### Q4. Which individual attributes contributed the most during the offer window?
 
The top three that contributed the most are as follows:
    1. Income
    2. Age
    3. Start year of membership
    
## Results
We also tried to build a model to predict the amount spent based on individual input and offer type.

Final results on the different trained models are:

| Model | R2 score |
| - | - |
| SVM Regressor | 0.041794040197025595 |
| Random Forest Regressor | -0.008060879289752076 |
| Random Forest Regressor with Grid Search | 0.059339282853018704 |

The final R2 score is better than that obtained with a SVM and with the default Random Forest. 

As part of an improvement task, I will try out PCA and SVM Regressor with better hyperparameters.

## Improvements

As part of an improvement task, we can try out PCA to find out newer dimenstions leading to exploring different customer segments based on the amount spent across each offer category. Comparing these distributions with the distributions we performed earlier, might give use much more information about which individuals to send different offer codes.

## Reference

[Starbucks Promotion Strategy — Capstone Project for Udacity’s Data Scientist Nanodegree](https://medium.com/@manojpatra/starbucks-promotion-strategy-capstone-project-for-udacitys-data-scientist-nanodegree-12031f8e8d29?sk=61c90f2d9d653e7ec457971b7efd7bd4)

[Project Rubric](https://review.udacity.com/#!/rubrics/2345/view)
