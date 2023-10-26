# Get lexical and semantic charateristics of all words in the stories
import json
import mne
import pandas as pd
import numpy as np
import nltk
import os
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
from wordfreq import zipf_frequency
from tqdm import tqdm
from matplotlib import pyplot as plt
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import math



def get_syntax(words_only):
    '''get syntactic category'''
    words_token = [word_tokenize(word) for word in words_only]
    words_tag = [nltk.pos_tag(word) for word in words_token]
    content_word_categories = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    words_fun = ['Content' if any(tag[1] in content_word_categories for tag in word_tags) else 'Functional' for word, word_tags in zip(words_only, words_tag)]
    return words_fun


## ---------------------------------------------------------------
# get lexical charateristics
## ---------------------------------------------------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
stories = os.listdir(my_path + 'stimuli/text/')
stories=['cable_spool_fort.txt', 'easy_money.txt', 'the_black_willow.txt']
for story in stories:
    fname = my_path + 'stimuli/text/' + story
    with open(fname,'r') as file:
        all_content = file.read().split()
    words_only = [word for word in all_content if word.isalpha()]
    words_len = [len(word) for word in words_only]
    words_freq = [zipf_frequency(word, "en") for word in words_only]
    words_token = [word_tokenize(word) for word in words_only]
    words_fun = get_syntax(words_only)
    
    # get lexical properties for all content words
    index_content = [i for i, word in enumerate(words_fun) if word == 'Content']
    content_words = [words_only[i] for i in index_content]
    content_words_len = [words_len[i] for i in index_content]
    content_words_freq = [words_freq[i] for i in index_content]
    df_lex = pd.DataFrame({'content_words': content_words,
                           'length': content_words_len,
                           'frequency': content_words_freq})
    
    # get word2vec values for all content words
    padded_words = ['xyzxyz']*5
    all_words= padded_words + content_words    
    missing_words = set([word for word in all_words if word not in model.key_to_index])
    similarities = []
    for i in range(5,len(all_words)):
        word_n_minus_1 = all_words[i - 1]
        word_n_minus_2 = all_words[i - 2]
        word_n_minus_3 = all_words[i - 3]
        word_n_minus_4 = all_words[i - 4]
        word_n_minus_5 = all_words[i - 5]
        word_n = all_words[i]

        if word_n in missing_words:
            similarity_n_1 = similarity_n_2 = similarity_n_3 = similarity_n_4 = similarity_n_5 = math.nan
        else:
            similarity_n_1 = model.similarity(word_n_minus_1, word_n) if word_n_minus_1 not in missing_words else math.nan
            similarity_n_2 = model.similarity(word_n_minus_2, word_n) if word_n_minus_2 not in missing_words else math.nan
            similarity_n_3 = model.similarity(word_n_minus_3, word_n) if word_n_minus_3 not in missing_words else math.nan
            similarity_n_4 = model.similarity(word_n_minus_4, word_n) if word_n_minus_4 not in missing_words else math.nan
            similarity_n_5 = model.similarity(word_n_minus_5, word_n) if word_n_minus_5 not in missing_words else math.nan
                    
        similarities.append([similarity_n_1, similarity_n_2, similarity_n_3, similarity_n_4, similarity_n_5])
        columns = ['similarity_n_1', 'similarity_n_2', 'similarity_n_3', 'similarity_n_4', 'similarity_n_5']
        similarity_df = pd.DataFrame(similarities, columns=columns)
    
    df_lexsem = pd.concat([df_lex, similarity_df], axis=1)  
    df_fname = my_path + 'stimuli/cloze/contentwords_lexsem_' + story.split('.')[0] + '.csv'
    df_lexsem.to_csv(df_fname, index=False)
    
    #get the list of words that can't be found in word2vec
    missing_words = [word for word in content_words if word not in model.key_to_index]
    words_to_write = ' '.join(missing_words)
    fname =  my_path + 'stimuli/cloze/w2vmissing_contentwords_' + story.split('.')[0] + '.txt'
    with open(fname, 'w') as f:
        f.write(words_to_write)