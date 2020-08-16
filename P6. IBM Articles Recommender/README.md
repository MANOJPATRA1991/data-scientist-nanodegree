# Recommendations with IBM

## Introduction

For this project we will analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles we think they will like.

The project is divided into the following tasks:

### I. Exploratory Data Analysis

Before making recommendations of any kind, we should explore the data for the project. Dive in to see what we can find.

### II. Rank Based Recommendations

To get started in building recommendations, we first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

### III. User-User Based Collaborative Filtering

In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users.

### IV. Content Based Recommendations (EXTRA - WORK IN PROGRESS)

Given the amount of content available for each article, there are a number of different ways in which someone might choose to implement a content based recommendations system. This is currently a work in progress, where I will be using some NLP techniques make this happen.

### V. Matrix Factorization

Finally, we have a machine learning approach to building recommendations. Using the user-item interactions, we build out a matrix decomposition. Using the decomposition, we get an idea of how well we can predict new articles an individual might interact with. We finally discuss which methods we might use moving forward, and how we might test how well our recommendations are working for engaging users.

### RUBRIC

[Project Rubric](https://review.udacity.com/#!/rubrics/2322/view)
