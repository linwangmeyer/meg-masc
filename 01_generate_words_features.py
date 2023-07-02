# RSA to words n, n+1 and n-1
import json
import mne
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
from wordfreq import zipf_frequency
from tqdm import tqdm
from matplotlib import pyplot as plt
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
from scipy.spatial.distance import pdist


def get_word_length(metadata):
    '''get word length'''
    metadata['Nr_letters'] = metadata['word'].apply(lambda x: len(x))
    return metadata


def get_syntax(metadata):
    '''get syntactic category'''
    metadata['tokens'] = metadata['word'].apply(lambda x: word_tokenize(x))
    metadata['pos_tags'] = metadata['tokens'].apply(lambda x: nltk.pos_tag(x))
    content_word_categories = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    metadata['word_category'] = metadata['pos_tags'].apply(lambda x: 'Content' if any(tag[1] in content_word_categories for tag in x) else 'Functional')
    return metadata


def get_wordfreq(metadata):
    '''get word frequency'''
    wfreq = lambda x: zipf_frequency(x, "en")
    metadata['word_freq'] = metadata['word'].apply(wfreq)
    return metadata


def get_word2vec(metadata):
    '''get word2vec representation for each word'''
    word_list = metadata['word'].tolist()
    word_vectors = []
    for word in word_list:
        try:
            vector = model[word]
            word_vectors.append(vector)
        except KeyError:
            word_vectors.append(np.nan)
    metadata['w2v'] = word_vectors
    #epochs.metadata = metadata
    #epochs = epochs[epochs.metadata['w2v'].notna()]
    return metadata


def find_cw_cloze(df_cloze,df_words):
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




def get_contents_epochs(epochs):
    '''get epochs with content words'''
    new_epochs = epochs[epochs.metadata['word_category']=='Content']
    return new_epochs



################################################################
# get word features for words
################################################################
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
#my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
task = 0
epochs_fname = my_path + f"/segments/sub{subject}_session{session}_task{task}.fif"
epochs = mne.read_epochs(epochs_fname)
epochs=epochs[epochs.metadata['story']=='lw1']

#get the metadata
df_words = epochs.metadata
df_words['word'] = df_words['word'].str.replace('It s', "It's")

# get cloze values
story = df_words['story'].unique()[0]
df_fname = my_path + 'stimuli/cloze/story_' + story + '_cloze.csv'
df_cloze = pd.read_csv(df_fname)
df_cloze['words'] = df_cloze['words'].astype(str).str.strip()
df_cw = find_cw_cloze(df_cloze,df_words)

# get other features
df_cw = get_word_length(df_cw)
df_cw = get_wordfreq(df_cw)
df_cw = get_syntax(df_cw)
#df_words = get_word2vec(df_words)

feature_fname = my_path + f"/segments/features_sub{subject}_session{session}_task{task}.csv"
df_cw.to_csv(feature_fname, index=False)



################################################################
## merge features with epoch metadata
################################################################
feature_fname = my_path + f"/segments/features_sub{subject}_session{session}_task{task}.csv"
df_cw = pd.read_csv(feature_fname)

epochs_fname = my_path + f"/segments/sub{subject}_session{session}_task{task}.fif"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
epochs=epochs[epochs.metadata['story']=='lw1']
epochs.metadata = df_cw

# get only content epochs
content_epochs = get_contents_epochs(epochs)
metadata = content_epochs.metadata
metadata['word_freq'].hist()
metadata['probs'].hist()

################################################################
# evoked activity to different conditions of features
################################################################
epochs_low = content_epochs[(metadata['probs'] < 0.4) & (metadata['probs'] > 0.1)]
epochs_high = content_epochs[metadata['probs'] >= 0.4]

evokeds = dict()
evokeds['low'] = epochs_low.average().apply_baseline((None, 0))
evokeds['high'] = epochs_high.average().apply_baseline((None, 0))
mne.viz.plot_compare_evokeds(evokeds, picks=['MEG 065'])

evoked_diff = mne.combine_evoked([evokeds['freq_low'], evokeds['freq_high']], weights=[1, -1])
evoked_diff.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag")

# check sensor layout
layout = mne.channels.find_layout(epochs.info, ch_type='meg')
layout.plot()
