'''Get top distribution for each context'''
import json
import mne
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Load the pre-trained LDA model
pretrained_model = LdaModel.load('path/to/pretrained_model')

# Load your own data
# Assuming you have a list of documents, where each document is a list of tokens
your_documents = [
    ['token1', 'token2', 'token3', ...],
    ['token4', 'token5', 'token6', ...],
    ...
]

# Create a dictionary mapping of word IDs to words
your_dictionary = Dictionary(your_documents)

# Convert your documents to a Bag-of-Words representation using the dictionary
your_corpus = [your_dictionary.doc2bow(doc) for doc in your_documents]

# Get the topic distributions for your documents
topic_distributions = pretrained_model[your_corpus]

# Print the topic distributions for each document
for i, doc_topics in enumerate(topic_distributions):
    print(f"Document {i+1}:")
    for topic_id, prob in doc_topics:
        print(f"Topic {topic_id}: {prob:.4f}")
    print()
