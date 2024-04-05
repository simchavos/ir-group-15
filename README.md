# Using Affect to Improve a Micro-Blogging Search Engine
This repository contains the code written for our report "*Using Affect to Improve a Micro-Blogging Search Engine*", written for the TU Delft MSc course Information Retrieval (IN4325).

## main.py
This is our main file. The main functions that are relevant are the following:

`train_model` trains our model. This function loads the training data, and then creates a SKLearn pipeline that allows us to predict the affect label of a sentence. Several pipelines are present and commented out. In the table below they are described:
| Model | Accuracy |
|--|--|
| Naive Bayes + CountVect | 0.88 (05) |
| SVM + TD-IDF | 0.88 (00) |
| Random Forest + TF-IDF | 0.87 (80) |
| Gradient Boosting + CountVect | 0.84 (25) |
| XGBoost + TF-IDF | 0.88 (55) |
| Logistic Regression + TF-IDF | 0.88 (05) |

The pipeline based on XGBoost + TF-IDF is uncommented and used by default.

`rank_tweets_...` calls the corresponding function from SentenceRanker.py. They are called with an argument specifying either the use of a cosine similarity ranker or BM25. By default, BM25 is selected.

`plot_jaccard_distance` uses MatPlotLib to plot the jaccard distance graphs.

`plot_jaccard_distance_loop` uses MatPlotLib that creates a jaccard distance graph of 100 random queries from the testing data.

## SentenceRanker.py
This file contains the implementation for a Cosine Similarity ranker and a BM25 implementation. The main functions that are  relevent are the following: 

`rank_sentences` ranks all sentences only on similarity to the query.

`rank_sentences_emotions_labels` ranks all sentences on similarity to the query and on the emotion score, based on labels given to sentences.

`rank_sentences_emotions_unlabeled` ranks all sentences on similarity to the query and on the emotion score, with unlabeled data. The emotion score is calculated for each sentence. 

These ranking functions work with either the Cosine Similarity ranker or BM25 depending on the arguments passed to them.

##  Data
The data our model is trained on can be found in the  `/data` folder. The data is split up into `test.csv`, `training.csv`, and `validation.csv`. For the training of the model, we have used `training.csv`. For finding the accuaracy of the models, we used `test.csv`. We did not use `validation.csv`. The data was originally scraped from the Twitter API by Saravia et al. for their publication [CARER: Contextualized Affect Representations for Emotion Recognition](https://aclanthology.org/D18-1404/). A part of this data was publically uploaded on [kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset).
