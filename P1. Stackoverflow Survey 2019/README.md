# Stackoverflow Survey 2019
The blog post can be found [here](https://medium.com/@manojpatra/stackoverflow-developer-survey-analysis-2019-cfbca09a088c).

## Motivation

Use CRISP-DM methodology to do analysis of the Stackoverflow survey data from 2019 to find out about the compensation, job satisfaction and tools used by developers across the industry.

## Running the project

1. Create a new virtual environment. Read more [here](https://docs.python.org/3/library/venv.html).
2. Activate the virtual environment.
3. Clone this repository into the folder where the virtual environment was created.
4. Create a folder named **data** inside the folder **P1. Stackoverflow Survey 2019**. This new folder should have four folders as mentioned below in the project structure - external, interim, processed, raw.
5. Download the Stackoverflow survey data from this [link](https://insights.stackoverflow.com/survey) and save the `.csv` files in the location `P1. Stackoverflow Survey 2019/data/external`
6. Run `pip install -r requirements.txt` to install the dependencies.
7. Run `jupyter notebook` to open the notebook

## Libraries used

### [sklearn](https://scikit-learn.org/stable/)

Free machine learning library for the Python programming language.

### [seaborn](https://seaborn.pydata.org/)

Python data visualization library based on Python.

### [matplotlib](https://matplotlib.org/)

Plotting library for the Python programming language and its numerical mathematics extension NumPy. 

### [numpy](https://numpy.org)

The fundamental package for scientific computing with Python.

### [pandas](https://pandas.pydata.org/)

pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

### [ipython](https://ipython.org/)

IPython is a command shell for interactive computing in multiple programming languages, originally developed for the Python programming language, that offers introspection, rich media, shell syntax, tab completion, and history. 

### [jupyter](https://jupyter.org/)

The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.

## Project structure

```
- P1. Stackoverflow Survey 2019
  |
  | - data
  |   |
  |   | - external - Contains external data
  |   |
  |   | - interim - Contains data as part of the ongoing process
  |   |
  |   | - processed - Contains final processed data
  |   |
  |   | - raw - Contains other raw data
  |   
  | - notebooks
  |   |
  |   | - 1.0-mkp-survey-results-public-data-exploration.ipynb - Exploration and analysis of the public response data. This file contains all of the analysis and findings.
  |   |
  |   | - 1.0-mkp-survey-results-schema-data-exploration.ipynb - Exploration and analysis of the schema data
  |
  | - references
  |   |
  |   | - README_2019.txt - Supporting document that provides information related to the Stackoverflow survey data
  |   |
  |   | - so_survey_2019.pdf - More details
  |
  | - reports
  |   |
  |   | - 1.0-mkp-survey-results-public-data-exploration.html - HTML generated from the notebook which serves as the final report
  
```

## Results of the analysis

### How satisfied are individuals with their jobs in terms of salary?

We found out that job satisfaction does become better with compensation. The effect is more seen for individuals in the salary range of 0 to 280000 USD.

### Which profession is the most preferred by individuals?

We found out that most of the work force consists of individuals experienced in the field of web developement, followed by application developement.

### Which profession is the most rewarding in terms of salary?

We found out that Engineering Manager and Senior executive/VP are the most paid individuals in terms of salary. 
Among individuals working in the field of data, Data Engineers are the most paid. 
Among individuals working in the field of web developement, full stack developers are the most paid.

### How does work load vary with compensation?

The individuals drawing higher salary do end up working more.

### Analysis of tools being used by different categories of developers

We created a function which gives a summary of frameworks/databases/languages, etc. being used by developers of different categories

## References

[1]. [StackOverflow Survey 2019](https://insights.stackoverflow.com/survey/2019)
[2]. [CRISP-DM](https://www.ibm.com/support/knowledgecenter/SS3RA7_sub/modeler_crispdm_ddita/clementine/crisp_help/crisp_overview_container.html)







