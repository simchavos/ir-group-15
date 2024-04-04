import sys
import math

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pyterrier as pt
import nltk
from enum import Enum

class SimMetric(Enum):
    COSINE = 1
    BM52 = 2
class SentenceRanker:
    def __init__(self, sentences):
        # nltk.download("punkt")
        self.sentences = sentences
        self.vectorizer = CountVectorizer().fit(sentences)
        self.vectors = self.vectorizer.transform(sentences)
        self.corpus = [sentence.lower().split() for sentence in sentences]
        self.bm25 = BM25(self.corpus)


    """rank all sentences only on similarity to the query"""
    def rank_sentences(self, query, simMetric = SimMetric.COSINE):
        query_vector = self.vectorizer.transform([query])
        if simMetric == SimMetric.COSINE:
            similarity_scores = cosine_similarity(query_vector, self.vectors)
            ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
            ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        elif simMetric == SimMetric.BM52:
            query = query.replace("'", " '").split()
            similarity_scores = self.bm25.score(query)
            result = np.array([sum(pair) for pair in zip(*similarity_scores)])
            ranked_sentences_indices = result.argsort()[::-1]
            ranked_sentences = [(self.sentences[idx], result[idx]) for idx in ranked_sentences_indices]
        return ranked_sentences

    """ rank all sentences on similarity to the query and on the emotion score, based on labels given to sentences.
        The emotion score is calculated for each sentence. The emotion score is the probability of the query belonging
        to the emotion corresponding to the label given to a sentence."""
    def rank_sentences_emotions_labels(self, query, emotions, labels, simMetric = SimMetric.COSINE):
        query_vector = self.vectorizer.transform([query])
        if simMetric == SimMetric.COSINE:
            similarity_scores = cosine_similarity(query_vector, self.vectors)
            for i,score in enumerate(similarity_scores[0]):
                label = labels[i]
                emotion_score = emotions[label]
                similarity_scores[0, i] *= emotion_score
            ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
            ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        elif simMetric == SimMetric.BM52:
            query = query.replace("'", " '").split()
            similarity_scores = self.bm25.score(query)
            result = np.array([sum(pair) for pair in zip(*similarity_scores)])
            for i,score in enumerate(result):
                label = labels[i]
                emotion_score = emotions[label]
                result[i] *= emotion_score
            ranked_sentences_indices = result.argsort()[::-1]
            ranked_sentences = [(self.sentences[idx], result[idx]) for idx in ranked_sentences_indices]

        return ranked_sentences

    """ rank all sentences on similarity to the query and on the emotion score, with unlabeled data
        The emotion score is calculated for each sentence. The emotion score is calculated by taking the probability
        vectors for emotions of both the query and a sentence, and taking the dot product of these 2 vectors."""
    def rank_sentences_emotions_unlabeled(self, query, emotions, labels, simMetric = SimMetric.COSINE):
        query_vector = self.vectorizer.transform([query])
        if simMetric == SimMetric.COSINE:
            similarity_scores = cosine_similarity(query_vector, self.vectors)
            for i, score in enumerate(similarity_scores[0]):
                label = labels[i]
                emotion_score = np.dot(label, emotions)
                similarity_scores[0, i] *= emotion_score
            ranked_sentences_indices = similarity_scores.argsort()[0][::-1]
            ranked_sentences = [(self.sentences[idx], similarity_scores[0, idx]) for idx in ranked_sentences_indices]
        if simMetric == SimMetric.BM52:
            query = query.replace("'", " '").split()
            similarity_scores = self.bm25.score(query)
            result = np.array([sum(pair) for pair in zip(*similarity_scores)])
            for i, score in enumerate(result):
                label = labels[i]
                emotion_score = np.dot(label, emotions)
                result[i] *= emotion_score
            ranked_sentences_indices = result.argsort()[::-1]
            ranked_sentences = [(self.sentences[idx], result[idx]) for idx in ranked_sentences_indices]

        return ranked_sentences





class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus) / len(corpus)
        self.idf = {}
        self.doc_len = [len(doc) for doc in corpus]
        self.doc_count = len(corpus)
        self.compute_idf()

    def compute_idf(self):
        for doc in self.corpus:
            for term in set(doc):
                if term not in self.idf:
                    self.idf[term] = 1
                else:
                    self.idf[term] += 1

        for term, doc_count in self.idf.items():
            self.idf[term] = math.log((self.doc_count - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def score(self, query):
        scores = np.zeros((len(query), len(self.corpus)))
        for i, term in enumerate(query):
            if term not in self.idf:
                continue
            for j, doc in enumerate(self.corpus):
                f = doc.count(term)
                score = self.idf[term] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
                scores[i, j] = score
        return scores