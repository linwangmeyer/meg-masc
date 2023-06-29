# RSA to words n, n+1 and n-1
import json
import mne
import pandas as pd
import numpy as np
import nltk
from wordfreq import zipf_frequency
import mne_rsa
from tqdm import tqdm
from matplotlib import pyplot as plt
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
from scipy.spatial.distance import pdist


def get_contents_epochs(epochs):
    '''get epochs with content words'''
    new_epochs = epochs[epochs.metadata['word_category']=='Content']
    return new_epochs


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

################################################################
## Run functions: get content words
################################################################
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
#my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
epochs_fname = my_path + f"/segments/session{session}_sub{subject}"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
epochs_content = get_contents_epochs(epochs)
epochs_fname = my_path + f"/segments/session{session}_sub{subject}-content-epochs.fif"
epochs_content.save(epochs_fname,overwrite=True)

################################################################
# get other word features
epochs_fname = my_path + f"/segments/session{session}_sub{subject}-content-epochs.fif"
epochs = mne.read_epochs(epochs_fname)
metadata = epochs.metadata

# get word frequency
metadata = get_wordfreq(metadata)
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


