# R-Gender_Tweet-Classifier
Naive-Bayes gender classifier of tweets

This repository contains an R script for performing gender classification of tweets using a Naive Bayes classifier. The script includes steps for data loading, preprocessing, model training, evaluation, and further analysis of the language used by different genders.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Script Description](#script-description)
- [Further Analysis](#further-analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to classify the gender of Twitter users based on the content of their tweets. It utilizes the Naive Bayes algorithm, a probabilistic classifier, to predict whether a tweet was likely authored by a male or female user. The script also explores differences in word usage between the classified genders through frequency analysis, topic modeling, and semantic network analysis.

## Installation

Before running the script, you need to have R installed on your system. You also need to install the following R packages. You can install them directly within R using the `install.packages()` command as shown in the script:

```R
install.packages("quanteda")
install.packages("textstem")
install.packages("quanteda.textmodels")
install.packages("caret")
install.packages("ggplot2")
install.packages("quanteda.textstats")
install.packages("stm")
install.packages("wordcloud")
install.packages("igraph")
install.packages("wordnet")
install.packages("compareDF")
install.packages("testthat")
Once installed, these packages will be loaded by the script using the library() command.

Usage
Clone the repository:

Bash

git clone [repository URL]
cd R-Gender_Tweet-Classifier
Prepare your data: Ensure your tweet data is in a CSV file named gender_classifier.csv (or modify the script accordingly). The expected data format is described in the Data Format section. Place this file in the /content directory (or adjust the setwd() command in the script).

Run the R script: Open R or RStudio, navigate to the repository directory, and execute the script:

R

source("your_script_name.R") # Replace "your_script_name.R" with the actual name of your R script
Alternatively, you can run the script line by line in an R interactive session.

View the results: The script will output classification results, statistical evaluations, word frequency analysis, topic models, and semantic network visualizations. These will be displayed in your R console or through generated plots. Community detection results for male and female networks will be saved as Communities_male.csv and Communities_female.csv in your working directory.

Data Format
The script expects a CSV file named gender_classifier.csv with the following columns (at a minimum):

X_unit_id: A unique identifier for each tweet.
text: The text content of the tweet.
X_golden: A boolean indicating if the annotation is golden (human-verified).
gender: The annotated gender of the tweet author (male, female, or brand). The script filters for male and female.
fav_number: The number of favorites the tweet received.
link_color: The link color used by the user.
gender.confidence: The confidence score of the gender annotation.
description: The user's profile description.
gender_gold: The golden gender annotation.
link_color: (Duplicate of link_color)
retweet_count: The number of retweets the tweet received.
sidebar_color: The sidebar color used by the user.
tweet_count: The total number of tweets by the user.
Note: The script specifically uses the text and gender columns for the Naive Bayes classification. The other columns are used for potential filtering and as document variables in the corpus.

Script Description
The R script performs the following main steps:

Package Installation and Loading: Installs and loads necessary R packages for text analysis, machine learning, and visualization.
Data Import: Reads the gender_classifier.csv file into an R data frame.
Corpus Creation: Creates a corpus object from the tweet text and relevant metadata.
Data Subsetting: Filters the corpus to include only tweets classified as "male" or "female" and further subsets based on a confidence threshold for the gender annotation.
Data Splitting: Splits the data into training (80%) and testing (20%) sets.
Document-Term Matrix (DTM) Creation: Converts the text data into a DTM, which represents the frequency of words in each tweet. This involves tokenization, lowercasing, stopword removal, and lemmatization.
Naive Bayes Model Training: Trains a Naive Bayes classification model using the training DTM and the gender labels.
Model Evaluation: Predicts gender on the test data and evaluates the model's performance using a confusion matrix and various statistical metrics.
Feature Importance: Identifies the words that are most indicative of male and female authors according to the model.
Classification of Other Datasets (Spam and Fake News): Demonstrates how the trained gender classification model can be applied to other text datasets (spam and fake news) for prediction, although the relevance of this classification might be limited.
Further Linguistic Analysis:
Relative Word Frequencies: Visualizes the relative frequencies of the top words and potentially gendered terms across male and female tweets.
Term Distance Clustering: Explores the semantic relationships between frequent words using hierarchical clustering.
Topic Modeling (STM): Identifies latent topics within the entire dataset and separately for male and female tweets.
Semantic Network Analysis (Co-occurrence Graphs): Creates and visualizes networks of co-occurring words in the entire dataset and separately for male and female tweets, allowing for the exploration of contextual differences.
Community Detection: Identifies clusters of closely connected words within the male and female semantic networks.
Further Analysis
The script includes sections for further linguistic analysis, such as exploring relative word frequencies, topic modeling, and semantic networks. You can modify these sections to investigate specific linguistic features or patterns of interest related to gendered language in tweets. The community detection results in the CSV files can be further analyzed to understand the different thematic clusters of words associated with each gender.

Contributing
Contributions to this project are welcome. Feel free to fork the repository, make changes, 1  and submit pull requests. Please ensure that your code is well-documented and follows good coding practices.   
