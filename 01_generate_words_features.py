# RSA to words n, n+1 and n-1
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
    df_lexsem: pre-obtained cloze values from the story'''
    all_words = df_lexsem['content_words'].tolist()
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
for file in file_lists[2:]:
    
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
    
    # get content words in df_cloze
    words_fun1 = get_syntax(df_cloze['words'].tolist())
    index_content1 = [i for i, word in enumerate(words_fun1) if word == 'Content']
    df_cloze_content = df_cloze.loc[index_content1]
    df_cloze_content.reset_index(drop=True,inplace=True)
    
    # get content words in df_words, remove special characters
    words_fun2 = get_syntax(df_words['word'].tolist())
    index_content2 = [i for i, word in enumerate(words_fun2) if word == 'Content']
    check_words = df_words['word'].tolist()
    index_sym = [i for i, word in enumerate(check_words) if word.isalpha()]
    
    index_remove = list(set(index_content2).intersection(index_sym))
    df_words_content = df_words.loc[index_remove]
    df_words_content.reset_index(drop=True,inplace=True)
    epochs_content = epochs[index_remove]
    epochs_content.metadata.reset_index(drop=True,inplace=True)
    
    # get cloze for content cws
    df_cw = find_cw_cloze(df_words_content,df_cloze_content)
    
    # check n_items and combine meta information
    if epochs_content.metadata.shape[0] != df_cw.shape[0]:
        raise ValueError('Mismatching number of trials between df_cloze and df_epochs')
    else:
        epochs_content.metadata = epochs_content.metadata.join(df_cw['probs'])
        epochs_content.metadata.reset_index(drop=True,inplace=True)
        df_meta = epochs_content.metadata
    
    # get lexico-semantic properties
    df_fname = my_path + 'stimuli/cloze/contentwords_lexsem_' + story[0].split('.')[0] + '.csv'
    df_lexsem = pd.read_csv(df_fname)
    df_words2 = epochs_content.metadata
    df_all = find_cw_lexsem(df_words2,df_lexsem)

    # check _items merge cloze and lexico-semantic properties
    if df_words2.shape[0] != df_all.shape[0]:
        raise ValueError('Mismatching number of trials between df_lexsem and df_epochs')
    else:
        df_merged = df_meta.join(df_all)
        df_merged.drop(['content_words'], axis=1, inplace=True)
    
    epochs_content.metadata = df_merged    
    epochs_fname = my_path + f"/segments_cw/{file}"
    epochs_content.save(epochs_fname,overwrite=True)
    
from difflib import ndiff
list1 = epochs_content.metadata['word'].str.lower().tolist()
list2 = df_cw['words'].str.lower().tolist()
list1_str = "\n".join(list1)
list2_str = "\n".join(list2)
differ = ndiff(list1_str.splitlines(), list2_str.splitlines())
missing_words = [line[2:] for line in differ if line.startswith('- ')]
missing_words


################################################################
## merge features with epoch metadata
################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
for file in file_lists:
    
    print(f'processing file: {file}')
    
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments_cw/{file}"
    epochs = mne.read_epochs(epochs_fname)

epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
metadata = epochs.metadata
metadata['frequency'].hist()
metadata['probs'].hist()

################################################################
# evoked activity to different conditions of features
################################################################
epochs_low = epochs[(metadata['probs'] < 0.4) & (metadata['probs'] > 0.1)]
epochs_high = epochs[metadata['probs'] >= 0.4]

evokeds = dict()
evokeds['low'] = epochs_low.average().apply_baseline((None, 0))
evokeds['high'] = epochs_high.average().apply_baseline((None, 0))

mne.viz.plot_compare_evokeds(evokeds, picks=['MEG 054'])
mne.viz.plot_compare_evokeds(evokeds, picks=['MEG 065'])

evoked_diff = mne.combine_evoked([evokeds['low'], evokeds['high']], weights=[1, -1])
evoked_diff.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag")

# check sensor layout
layout = mne.channels.find_layout(epochs.info, ch_type='meg')
layout.plot()
