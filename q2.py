import os
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess_text import preprocess_word

def calculate_tfidf(words):
    vectorizer = TfidfVectorizer()
    matrix_tfidf = vectorizer.fit_transform(words)
    names_of_features = vectorizer.get_feature_names_out()
    return matrix_tfidf, names_of_features



csv_file = "dataset.csv"
output_tfidf_file = "Q2.pkl"
df = pd.read_csv(csv_file)
print("Preprocessing word...............")
preprocessed_words = df['Text'].apply(lambda x: preprocess_word(str(x)))
print("Calculating TF-IDF scores............")
matrix_tfidf = calculate_tfidf(preprocessed_words)
with open('cleaned_texts.pkl', 'wb') as file:
    pickle.dump(preprocessed_words, file)

with open('tf_idf_results.pkl', 'wb') as file:
    pickle.dump(matrix_tfidf, file)