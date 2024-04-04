import csv
import sys
import math
import random

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# from bm25model import OkapiBM25Model
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.corpora import Dictionary
from SentenceRanker import SentenceRanker
from SentenceRanker import SimMetric
from scipy import stats
import pyterrier as pt
from pathlib import Path
from itertools import chain


# class BM25:
#     def __init__(self, corpus, k1=1.5, b=0.75):
#         self.corpus = corpus
#         self.k1 = k1
#         self.b = b
#         self.avgdl = sum(len(doc) for doc in corpus) / len(corpus)
#         self.idf = {}
#         self.doc_len = [len(doc) for doc in corpus]
#         self.doc_count = len(corpus)
#         self.compute_idf()
#
#     def compute_idf(self):
#         for doc in self.corpus:
#             for term in set(doc):
#                 if term not in self.idf:
#                     self.idf[term] = 1
#                 else:
#                     self.idf[term] += 1
#
#         for term, doc_count in self.idf.items():
#             self.idf[term] = math.log((self.doc_count - doc_count + 0.5) / (doc_count + 0.5) + 1)
#
#     def score(self, query):
#         scores = [0] * len(self.corpus)
#         for term in query:
#             if term not in self.idf:
#                 continue
#             for i, doc in enumerate(self.corpus):
#                 f = doc.count(term)
#                 scores[i] += self.idf[term] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
#         return scores



training_tweets = list()
training_labels = list()
test_tweets = list()
test_labels = list()
validation_tweets = list()
validation_labels = list()
predicted_labels = []


def train_model():
    with open("data/training.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            tweet, label = row[0], int(row[1])
            training_tweets.append(tweet)
            training_labels.append(label)

    with open("data/test.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            tweet, label = row[0], int(row[1])
            test_tweets.append(tweet)
            test_labels.append(label)

    with open("data/validation.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            tweet, label = row[0], int(row[1])
            validation_tweets.append(tweet)
            validation_labels.append(label)  # Assuming `data` is your dataset

    # Create a pipeline with a TF-IDF vectorizer and a Logistic Regression classifier
    pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(C=1.2, class_weight='balanced'))

    # Train the model
    pipeline.fit(training_tweets, training_labels)

    # Evaluate the model on the test set
    accuracy = pipeline.score(test_tweets, test_labels)
    print(f'Model accuracy: {accuracy}')
    return pipeline

def find_tweets_with_same_label(label):
    tweets = training_tweets + test_tweets + validation_tweets
    labels = training_labels + test_labels + validation_labels

    return [s for s, l in zip(tweets, labels) if l == label]



# def runBM25(query):
#     tweets = training_tweets + test_tweets + validation_tweets
#     # print(tweets)
#     split_tweets = [tweet.split() for tweet in tweets]
#     # print(split_tweets)
#     dictionary = Dictionary(split_tweets)  # fit dictionary
#     # print(dictionary)
#     corpus = [dictionary.doc2bow(line) for line in split_tweets]
#
#     bm25 = OkapiBM25Model(corpus=corpus, dictionary=dictionary)
#
#
#
#     print("start getting document scores")
#     document_scores = bm25[query]
#     print("start ranking")
#
#     ranked_documents = sorted(enumerate(document_scores), key=lambda x: x[1], reverse=True)
#     print(ranked_documents)
#     print("OK")
#     sys.exit(0)
#
#     # Calculate BM25 scores for documents
#     # scores = bm25.get_scores(query, tweets)
#     #
#     # ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#     # ranked_documents = [tweets[i] for i in ranked_indices]
#     #
#     # print("Ranked Documents:")
#     # for i, document in enumerate(ranked_documents):
#     #     if i >10:
#     #         break
#     #     print(document)

def rankTweetsEmotionless(query):
    tweets = training_tweets

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences(query)

    # print("_________________________________________________________________________________")
    # print("Without emotion: ")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >= 10:
    #         break
    return ranked_results

def rankTweetsLabels(query):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets
    labels = training_labels

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_labels(query,emotion,labels)
    #
    # print("_________________________________________________________________________________")
    # print("With labeled emotions: ")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >= 10:
    #         break
    return ranked_results


def rankTweetsUnlabeled(query):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets

    if predicted_labels == []:
        for tweet in tweets:
            predicted_labels.append(trained_pipeline.predict_proba([tweet])[0])

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_unlabeled(query, emotion, predicted_labels)

    # print("_________________________________________________________________________________")
    # print("With unlabeled emotions")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >=10:
    #         break
    return ranked_results









def rankTweetsEmotionlessBM25(query):
    tweets = training_tweets

    ranker = SentenceRanker(tweets, )
    ranked_results = ranker.rank_sentences(query, simMetric=SimMetric.BM52)

    # print("_________________________________________________________________________________")
    # print("Without emotion: ")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >= 10:
    #         break
    return ranked_results

def rankTweetsLabelsBM25(query):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets
    labels = training_labels

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_labels(query,emotion,labels,simMetric=SimMetric.BM52)
    #
    # print("_________________________________________________________________________________")
    # print("With labeled emotions: ")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >= 10:
    #         break
    return ranked_results


def rankTweetsUnlabeledBM25(query):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets

    if predicted_labels == []:
        for tweet in tweets:
            predicted_labels.append(trained_pipeline.predict_proba([tweet])[0])

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_unlabeled(query, emotion, predicted_labels,simMetric=SimMetric.BM52)

    # print("_________________________________________________________________________________")
    # print("With unlabeled emotions")
    # for i, (sentence, similarity_score) in enumerate(ranked_results):
    #     print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
    #     if i >=10:
    #         break
    return ranked_results

def remove_score(list):
    newList = []
    for (sentence, score) in list:
        newList.append(sentence)
    return newList

def plotJaccardDistance(scoreList1, scoreList2,nameGraph):
    list1 = remove_score(scoreList1)
    list2 = remove_score(scoreList2)

    distance_vs_percentage =[]
    for numberOfSamples in range(1,20):
    # for percentage in [i * 0.005 for i in range(1, 101)]:
    #     numberOfSamples = int(len(list1) * percentage)
        score1 = scoreList1[numberOfSamples][1]
        score2 = scoreList2[numberOfSamples][1]
        if(score1 == 0 or score2 == 0):
            break
        currentList1 = list1[0:numberOfSamples]
        currentList2 = list2[0:numberOfSamples]
        intersection = list(set(currentList1) & set(currentList2))
        union = list(set(currentList1) | set(currentList2))
        if len(union) ==0:
            jaccardDistance =0
        else:
            jaccardDistance = len(intersection) / len(union)
        distance_vs_percentage.append((numberOfSamples, jaccardDistance))

    x_values = [pair[0] for pair in distance_vs_percentage]
    y_values = [pair[1] for pair in distance_vs_percentage]
    # Plot the pairs
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    # Add labels and title
    plt.xlabel('first x results considered')
    plt.ylabel('Jaccard distance')
    plt.title(nameGraph)
    # Display the plot
    plt.grid(True)
    plt.show()






def plotJaccardDistanceLoop(scoreList1, scoreList2, nameGraph):
    # distance_vs_percentage_aggregate = []
    average_distances = {numberOfSamples: [] for numberOfSamples in range(1, 21)}
    sublist = random.sample(test_tweets, 100)
    for i, tweet in enumerate(sublist):
        print(i)
        # if i >3:
        #     break
        scoreList1 = rankTweetsEmotionlessBM25(tweet)
        scoreList2 = rankTweetsLabelsBM25(tweet)

        list1 = remove_score(scoreList1)
        list2 = remove_score(scoreList2)

        # Calculate Jaccard distance for each numberOfSamples
        for numberOfSamples in range(1, 21):
            score1 = scoreList1[numberOfSamples][1]
            score2 = scoreList2[numberOfSamples][1]
            if score1 == 0 or score2 == 0:
                break
            currentList1 = list1[:numberOfSamples]
            currentList2 = list2[:numberOfSamples]
            intersection = list(set(currentList1) & set(currentList2))
            union = list(set(currentList1) | set(currentList2))
            if len(union) == 0:
                jaccardDistance = 0
            else:
                jaccardDistance = len(intersection) / len(union)
            average_distances[numberOfSamples].append(jaccardDistance)

        # Calculate the average Jaccard distance for each numberOfSamples
        distance_vs_percentage_aggregate = [sum(distances) / len(distances) if distances else 0 for
                                            numberOfSamples, distances in average_distances.items()]


    print(distance_vs_percentage_aggregate)
    x_values = list(range(1, 21))
    y_values = distance_vs_percentage_aggregate
    # Plot the pairs
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    # Add labels and title
    plt.xlabel('first x results considered')
    plt.ylabel('Jaccard distance')
    plt.title("emotionless vs labeled emotions for BM25")
    # Display the plot
    plt.grid(True)
    plt.show()


# def testRanking():
#     # if not pt.started():
#     #     pt.init(tqdm="notebook")
#
#     dataset = training_tweets
#
#     index = pt.index.IterDictIndexer(
#         str(Path.cwd()),  # this will be ignored
#         meta={
#             "sentence": 65536,  # Assuming 65536 is the maximum length of a sentence
#         },
#         type=pt.index.IndexingType.MEMORY,
#     ).index(dataset, fields=["sentence"])
#
#     first_stage_bm25 = pt.BatchRetrieve(
#         index,
#         wmodel="BM25",
#         num_results=100,
#         metadata=["sentence"],
#     )
#
#     print("Succes")
#     sys.exit(0)

def bm25Rannking(query):
    # emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets
    ranker = SentenceRanker(tweets)
    emotion = trained_pipeline.predict_proba([query])[0]
    labels = training_labels


    # corpus = [sentence.lower().split() for sentence in tweets]
    # query = query.replace("'", " '").split()
    # print(query)
    # bm25 = BM25(corpus)
    # scores = bm25.score(query)
    # print(scores)
    ranked_results = ranker.rank_sentences_emotions_labels(query, emotion,labels)
    print(ranked_results[:10])
    print("_______________________")
    ranked_results = ranker.rank_sentences(query, simMetric=SimMetric.BM52)
    print(ranked_results[:10])
    ranked_results = ranker.rank_sentences_emotions_labels(query, emotion,labels, simMetric=SimMetric.BM52)
    print(ranked_results[:10])

    if predicted_labels == []:
        for tweet in tweets:
            predicted_labels.append(trained_pipeline.predict_proba([tweet])[0])
    ranked_results = ranker.rank_sentences_emotions_unlabeled(query, emotion, predicted_labels, simMetric=SimMetric.BM52)
    print(ranked_results[:10])
    return ranked_results



    # ranker = SentenceRanker(tweets)
    # ranked_results = ranker.rank_sentences_emotions_unlabeled(query, emotion, predicted_labels)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    trained_pipeline = train_model()
    plotJaccardDistanceLoop([],[],"jo")
    # sys.exit(0)
    while True:
        query = input('Enter query: ')
        # emotion = trained_pipeline.predict([query])

        # print(f'Emotion: {emotions}')
        results_emotionless = rankTweetsEmotionlessBM25(query)
        results_labeled_emotions = rankTweetsLabelsBM25(query)
        results_unlabeled_emotion = rankTweetsUnlabeledBM25(query)

        plotJaccardDistance(results_emotionless, results_labeled_emotions, "emotionless vs labeled emotions")
        plotJaccardDistance(results_emotionless, results_unlabeled_emotion, "emotionless vs unlabeled emotion")
        plotJaccardDistance(results_labeled_emotions, results_unlabeled_emotion, "labeled emotions vs unlabeled emotions")

        # tweets_with_same_label = find_tweets_with_same_label(emotion)
        # print(tweets_with_same_label)
