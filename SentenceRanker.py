import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pyterrier as pt
import nltk


class SentenceRanker:
    def __init__(self, sentences):
        # nltk.download("punkt")
        self.sentences = sentences
        self.vectorizer = CountVectorizer().fit(sentences)
        self.vectors = self.vectorizer.transform(sentences)

    """rank all sentences only on similarity to the query"""
    def rank_sentences(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.vectors)
        ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
        ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        return ranked_sentences

    """ rank all sentences on similarity to the query and on the emotion score, based on labels given to sentences.
        The emotion score is calculated for each sentence. The emotion score is the probability of the query belonging
        to the emotion corresponding to the label given to a sentence."""
    def rank_sentences_emotions_labels(self, query, emotions, labels):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.vectors)
        for i,score in enumerate(similarity_scores[0]):
            label = labels[i]
            emotion_score = emotions[label]
            similarity_scores[0, i] *= emotion_score
        ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
        ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        return ranked_sentences

    """ rank all sentences on similarity to the query and on the emotion score, with unlabeled data
        The emotion score is calculated for each sentence. The emotion score is calculated by taking the probability
        vectors for emotions of both the query and a sentence, and taking the dot product of these 2 vectors."""
    def rank_sentences_emotions_unlabeled(self, query, emotions, labels):
        query_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.vectors)
        for i, score in enumerate(similarity_scores[0]):
            label = labels[i]
            emotion_score = np.dot(label, emotions)
            similarity_scores[0, i] *= emotion_score
        ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
        ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        return ranked_sentences