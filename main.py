import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

training_tweets = list()
training_labels = list()
test_tweets = list()
test_labels = list()
validation_tweets = list()
validation_labels = list()


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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trained_pipeline = train_model()
    while True:
        query = input('Enter query: ')
        emotion = trained_pipeline.predict([query])
        print(f'Emotion: {emotion}')
        tweets_with_same_label = find_tweets_with_same_label(emotion)
        print(tweets_with_same_label)
