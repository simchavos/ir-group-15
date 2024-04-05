import csv
import random

from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from SentenceRanker import SentenceRanker
from SentenceRanker import SimMetric

training_tweets = list()
training_labels = list()
test_tweets = list()
test_labels = list()
validation_tweets = list()
validation_labels = list()
predicted_labels = []

# Trains our model
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
            validation_labels.append(label)

    # Different models. Comment at the end of each line is the accuracy we found.
    #pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(C=1.2, class_weight='balanced')) #0.8805
    #pipeline = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB()) #0.8805
    #pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), SVC(kernel='linear', class_weight='balanced')) #0.88
    #pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), RandomForestClassifier(n_estimators=100, class_weight='balanced')) #0.878
    #pipeline = make_pipeline(CountVectorizer(stop_words='english'), GradientBoostingClassifier(n_estimators=100)) #0.8425
    pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), XGBClassifier(use_label_encoder=False, eval_metric='logloss')) #0.8855

    # Train the model
    pipeline.fit(training_tweets, training_labels)

    # Evaluate the model on the test set
    accuracy = pipeline.score(test_tweets, test_labels)
    print(f'Model accuracy: {accuracy}')
    return pipeline

# returns all tweets with a specific label
def find_tweets_with_same_label(label):
    tweets = training_tweets + test_tweets + validation_tweets
    labels = training_labels + test_labels + validation_labels
    return [s for s, l in zip(tweets, labels) if l == label]

# print the first 10 ranked results and their similarity scores
def print_results(type, ranked_results):
    print("_________________________________________________________________________________")
    print(type)
    for i, (sentence, similarity_score) in enumerate(ranked_results):
        print(f"{sentence} (Similarity Score: {similarity_score:.4f})")
        if i >= 10:
            break

def rank_tweets_emotionless(query, ranking):
    tweets = training_tweets
    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences(query, simMetric=ranking)
    print_results("Without emotion", ranked_results)
    return ranked_results

def rank_tweets_labels(query, ranking):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets
    labels = training_labels
    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_labels(query, emotion, labels, simMetric=ranking)
    print_results("With labeled emotion", ranked_results)
    return ranked_results

def rank_tweets_unlabeled(query, ranking):
    emotion = trained_pipeline.predict_proba([query])[0]
    tweets = training_tweets

    if predicted_labels == []:
        for tweet in tweets:
            predicted_labels.append(trained_pipeline.predict_proba([tweet])[0])

    ranker = SentenceRanker(tweets)
    ranked_results = ranker.rank_sentences_emotions_unlabeled(query, emotion, predicted_labels, simMetric=ranking)
    print_results("With unlabeled emotion", ranked_results)
    return ranked_results

def remove_score(list):
    newList = []
    for (sentence, score) in list:
        newList.append(sentence)
    return newList

def plot_jaccard_distance(scoreList1, scoreList2, nameGraph):
    list1 = remove_score(scoreList1)
    list2 = remove_score(scoreList2)
    distance_vs_percentage = []

    for numberOfSamples in range(1,20):
        score1 = scoreList1[numberOfSamples][1]
        score2 = scoreList2[numberOfSamples][1]
        if (score1 == 0 or score2 == 0):
            break
        currentList1 = list1[0:numberOfSamples]
        currentList2 = list2[0:numberOfSamples]
        intersection = list(set(currentList1) & set(currentList2))
        union = list(set(currentList1) | set(currentList2))
        if len(union) == 0:
            jaccardDistance = 0
        else:
            jaccardDistance = len(intersection) / len(union)
        distance_vs_percentage.append((numberOfSamples, jaccardDistance))

    x_values = [pair[0] for pair in distance_vs_percentage]
    y_values = [pair[1] for pair in distance_vs_percentage]

    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.xlabel('first x results considered')
    plt.ylabel('Jaccard distance')
    plt.title(nameGraph)
    plt.grid(True)
    plt.show()

def plot_jaccard_distance_loop(scoreList1, scoreList2):
    average_distances = {numberOfSamples: [] for numberOfSamples in range(1, 21)}
    sublist = random.sample(test_tweets, 100)

    for i, tweet in enumerate(sublist):
        print(i)
        scoreList1 = rank_tweets_emotionless(tweet, SimMetric.BM52)
        scoreList2 = rank_tweets_labels(tweet, SimMetric.BM52)

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

    x_values = list(range(1, 21))
    y_values = distance_vs_percentage_aggregate

    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.xlabel('first x results considered')
    plt.ylabel('Jaccard distance')
    plt.title("emotionless vs labeled emotions for BM25")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    trained_pipeline = train_model()
    #plot_jaccard_distance_loop([], [])
    while True:
        query = input('Enter query: ')
        emotion = trained_pipeline.predict([query])
        print(f'Query label: {emotion}')

        results_emotionless = rank_tweets_emotionless(query, SimMetric.BM52)
        results_labeled_emotions = rank_tweets_labels(query, SimMetric.BM52)
        results_unlabeled_emotion = rank_tweets_unlabeled(query, SimMetric.BM52)

        #plot_jaccard_distance(results_emotionless, results_labeled_emotions, "emotionless vs labeled emotions")
        #plot_jaccard_distance(results_emotionless, results_unlabeled_emotion, "emotionless vs unlabeled emotion")
        #plot_jaccard_distance(results_labeled_emotions, results_unlabeled_emotion, "labeled emotions vs unlabeled emotions")