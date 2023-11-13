# get all features for each word
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


def find_cw_cloze_old(df_words,df_cloze):
    '''Get the cloze values for the remaining items
    df_words: dataframe that contains remaining words in the epochs
    df_cloze: pre-obtained cloze values from the story'''
    rows = []
    last_index = 0  # Initialize the last index found
    for index_word, row_word in df_words.iterrows():
        found = False
        for index_cloze, row_cloze in df_cloze.loc[last_index:, :].iterrows():
            if row_cloze['words'].lower() == row_word['word'].lower():
                rows.append(row_cloze)
                last_index = index_cloze + 1  # Update the last index found
                found = True
                break
        if not found:
            print(f"Word '{row_word['word']}' not found in DataFrame.")
    new_df = pd.DataFrame(rows)
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def find_cw_cloze_old2(df_cloze,df_words):
    df_words.reset_index(drop=True,inplace=True)
    rows = []
    last_index = 0  # Initialize the last index found
    for index_word, row_word in df_words.iterrows():
        found = False
        for index_cloze, row_cloze in df_cloze.loc[last_index:, :].iterrows():
            if row_cloze['words'].lower() == row_word['word'].lower():
                rows.append(row_cloze)
                last_index = index_cloze + 1  # Update the last index found
                found = True
                break
        if not found:
            print(f"Word '{row_word['word']}' not found in DataFrame.")
    new_df = pd.DataFrame(rows)
    new_df.reset_index(drop=True, inplace=True)
    df = pd.concat([new_df,df_words],axis=1)
    df = df.drop(['words','story','story_uid','sound_id','sound','onset'],axis=1)
    df = df.reindex(columns=['word', 'word_index', 'duration', 'probs'])
    return df


def find_cw_cloze(df_words,df_cloze):
    '''Get the cloze values for the remaining items
    df_words: dataframe that contains remaining words in the epochs
    df_cloze: pre-obtained cloze values from the story'''
    all_words = df_cloze['words'].tolist()
    new_words = df_words['word'].tolist()
    extra_word_indices = []
    k=0
    for i, word in enumerate(all_words):
        if word in new_words[k:i+1]:
            k=k+1
        elif word not in new_words[k:i + 1]:
            extra_word_indices.append(i)
    all_indices = range(len(all_words))
    kept_indices = [index for index in all_indices if index not in extra_word_indices]
    df_kept = df_cloze.loc[kept_indices]
    df_kept.reset_index(drop=True, inplace=True)
    return df_kept


def find_cw_lexsem(df_words,df_lexsem):
    '''Get the lexico-semantic values for the critical items
    df_words: dataframe that contains remaining words in the epochs
    df_lexsem: pre-obtained lexical values from the story'''
    all_words = df_lexsem['content_words'].tolist()
    new_words = df_words['words'].tolist()
    extra_word_indices = []
    k=0
    for i, word in enumerate(all_words):
        if word in new_words[k:i+1]:
            k=k+1
        elif word not in new_words[k:i + 1]:
            extra_word_indices.append(i)
    all_indices = range(len(all_words))
    kept_indices = [index for index in all_indices if index not in extra_word_indices]
    df_kept = df_lexsem.loc[kept_indices]
    df_kept.reset_index(drop=True, inplace=True)
    return df_kept


def get_syntax(words_only):
    '''get syntactic category'''
    words_token = [word_tokenize(word) for word in words_only]
    words_tag = [nltk.pos_tag(word) for word in words_token]
    content_word_categories = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    words_fun = ['Content' if any(tag[1] in content_word_categories for tag in word_tags) else 'Functional' for word, word_tags in zip(words_only, words_tag)]
    return words_fun


################################################################
# get word features for words
################################################################
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
for file in file_lists:
    
    print(f'processing file: {file}')
    
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    epochs.metadata.reset_index(drop=True,inplace=True)
    df_words = epochs.metadata
    
    # get the story name
    story = epochs.metadata['story'].unique()
    
    # get cloze values
    df_fname = my_path + 'stimuli/cloze/cloze_FullContext_' + story[0].split('.')[0] + '.csv'
    df_cloze = pd.read_csv(df_fname)
    df_cloze['words'] = df_cloze['words'].astype(str).str.strip()
    
    # get cloze for all cws
    df_cw = find_cw_cloze(df_words,df_cloze)
    
    # get content words in df_cloze
    words_fun1 = get_syntax(df_cw['words'].tolist())
    index_content1 = [i for i, word in enumerate(words_fun1) if word == 'Content']
    df_cloze_content = df_cw.loc[index_content1]
    df_cloze_content.reset_index(drop=True,inplace=True)
    
    # get epochs with content words
    epochs = epochs[index_content1]
    epochs.metadata.reset_index(inplace=True)
    df_merged = epochs.metadata.join(df_cloze_content)
    df_merged.drop(columns=['index','word'], inplace=True)    
    epochs.metadata = df_merged
    epochs.metadata.reset_index(inplace=True)
    epochs.metadata.drop(columns=['index'], inplace=True) 
    
    # get other lexical properties
    df_fname = my_path + 'stimuli/cloze/contentwords_lexsem_' + story[0].split('.')[0] + '.csv'
    df_lexsem = pd.read_csv(df_fname)    
    df_cw_lexsem = find_cw_lexsem(df_merged,df_lexsem)
    df_all = epochs.metadata.join(df_cw_lexsem)
    df_all.drop(columns=['content_words'], inplace=True)    
    epochs.metadata = df_all
    epochs.metadata.reset_index(inplace=True)
    epochs.metadata.drop(columns=['index'], inplace=True) 
    
    epochs.save(epochs_fname,overwrite=True)

