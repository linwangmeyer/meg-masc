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


def find_cloze_cw(df_cloze_all,df_words):
    '''match the cloze values with the cws identified in the MEG data
    df_cloze_all: cloze values for every word and punctuations in the text data
    df_words: metadata['words'] associated with the MEG data'''
    cw_lists = df_words['word'].to_list()
    for i, word in enumerate(cw_lists):
        #print(i)
        #print(word)
        while df_cloze_all.loc[i, 'words'] != word:
            pop_ele = df_cloze_all.loc[i, 'words']
            print(f'pop out {pop_ele}')
            df_cloze_all = df_cloze_all.drop(i, axis=0)
            df_cloze_all.reset_index(drop=True, inplace=True)
    return df_cloze_all



def get_contents_epochs(epochs):
    '''get epochs with content words'''
    new_epochs = epochs[epochs.metadata['word_category']=='Content']
    return new_epochs



################################################################
## get cloze value for every word, and match it with epochs
################################################################
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
#my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
task = 0
epochs_fname = my_path + f"/segments/sub{subject}_session{session}_task{task}.fif"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction

    
       


################################################################
# get other word features for all words
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
#my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
task = 0
word_fname = my_path + f"/segments/words_sub{subject}_session{session}_task{task}.csv"
df_words = pd.read_csv(word_fname)

# get features
df_words = get_word_length(df_words)
df_words = get_wordfreq(df_words)
df_words = get_syntax(df_words)
#df_words = get_word2vec(df_words)

epochs_fname = my_path + f"/segments/sub{subject}_session{session}_task{task}.fif"
epochs = mne.read_epochs(epochs_fname)
story = epochs.metadata['story'].to_list()[0]

df_fname = my_path + 'stimuli/cloze/story_' + story.lower() + '_cloze.csv'
df_cloze_all = pd.read_csv(df_fname)
df_cloze_cw = find_cloze_cw(df_cloze_all,df_words)

df_features = df_words.merge(df_cloze_cw, on='word', how='inner')
df_features = df_features.drop('word', axis=1)
feature_fname = my_path + f"/segments/features_sub{subject}_session{session}_task{task}.csv"
df_features.to_csv(feature_fname, index=False)


##########################################
## merge features with epoch metadata
feature_fname = my_path + f"/segments/features_sub{subject}_session{session}_task{task}.csv"
df_features = pd.read_csv(feature_fname)

epochs_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}.fif"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
metadata = epochs.metadata
metadata = metadata.merge(df_features, on = 'word', how = 'inner')
epochs.metadata = metadata


# get only content epochs
content_epochs = get_contents_epochs(epochs)

metadata['word_freq'].hist()


################################################################
# evoked activity to word frequency
epochs_low = epochs[metadata['word_freq'] < 5]
epochs_high = epochs[metadata['word_freq'] >= 5]

evokeds = dict()
evokeds['freq_low'] = epochs_low.average().apply_baseline((None, 0))
evokeds['freq_high'] = epochs_high.average().apply_baseline((None, 0))
mne.viz.plot_compare_evokeds(evokeds, picks=['MEG 065'])

evoked_diff = mne.combine_evoked([evokeds['freq_low'], evokeds['freq_high']], weights=[1, -1])
evoked_diff.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag")

# check sensor layout
layout = mne.channels.find_layout(epochs.info, ch_type='meg')
layout.plot()

################################################################
# separate stories
story_names = epochs.metadata['story'].unique().tolist()

# get predictability


