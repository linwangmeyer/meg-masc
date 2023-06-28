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

def get_wordfreq(epochs):
    '''get word frequency'''
    wfreq = lambda x: zipf_frequency(x, "en")
    epochs.metadata['word_freq'] = epochs.metadata['word'].apply(wfreq)
    return epochs

def get_contents(epochs):
    '''get epochs with content words'''
    new_epochs = epochs[epochs.metadata['word_category']=='Content']
    return new_epochs


def get_consecutive_contents(epochs, batch_size):
    '''get epochs with three content words in a row'''
    filtered_epochs_n = []
    filtered_metadata_nminus = []
    filtered_metadata_nplus = []
    
    num_epochs = len(epochs)
    num_batches = num_epochs // batch_size + 1
    
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_epochs)
        
        for i in range(start_idx + 1, end_idx - 1):
            cat_i = epochs[i].metadata['word_category']
            cat_i_minus = epochs[i-1].metadata['word_category']
            cat_i_plus = epochs[i+1].metadata['word_category']
            
            if cat_i.item() == 'Content' and cat_i_minus.item() == 'Content' and cat_i_plus.item() == 'Content':
                filtered_epochs_n.append(epochs[i])
                filtered_metadata_nminus.append(epochs[i-1].metadata)
                filtered_metadata_nplus.append(epochs[i+1].metadata)
    
    epochs_n = mne.concatenate_epochs(filtered_epochs_n)
    metadata_nminus = pd.concat(filtered_metadata_nminus, axis=0)
    metadata_nplus = pd.concat(filtered_metadata_nplus, axis=0)
    
    return epochs_n, metadata_nminus, metadata_nplus


def get_word2vec(epochs):
    '''get word2vec representation for each word'''
    word_list = epochs.metadata['word'].tolist()
    word_vectors = []
    for word in word_list:
        try:
            vector = model[word]
            word_vectors.append(vector)
        except KeyError:
            word_vectors.append(np.nan)
    epochs.metadata['w2v'] = word_vectors
    epochs = epochs[epochs.metadata['w2v'].notna()]
    return epochs


def Model_DSM(select_epochs,var=str):
    '''get pairwise similarity for a variable of interest'''
    metadata = select_epochs.metadata
    if len(metadata[var].values[0].shape) == 0:
        num_letters = metadata[var].values
        diff_matrix = np.abs(np.subtract.outer(num_letters, num_letters))
        indices = np.triu_indices(len(num_letters), k=1)
        higher_diagonal = diff_matrix[indices]
        dsm = higher_diagonal.flatten()
    elif len(metadata[var].values[0].shape) == 1:    
        word2vec_array = metadata[var].values
        word2vec_matrix = np.vstack(word2vec_array)
        dsm = pdist(word2vec_matrix, metric='cosine')
    else:
        dsm=None   
    return dsm



def generate_meg_dsms(select_epochs):
    '''calculate MEG data RDMs using spatial information: time*trialpair'''
    meg_data = select_epochs.get_data()
    n_trials, n_sensors, n_times = meg_data.shape
    data_dsm = []
    for i in range(n_times):
        dsm = mne_rsa.compute_dsm(meg_data[:, :, i], metric='correlation')
        data_dsm.append(dsm)
    data_dsm = np.array(data_dsm)
    return data_dsm



my_path = my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
epochs_fname = my_path + f"/Segments/session{session}_sub{subject}"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
epochs_n, metadata_nminus, metadata_nplus = get_consecutive_contents(epochs) #filter epochs: only keep trials with three consecutive content words
epochs_fname = my_path + f"/Segments/session{session}_sub{subject}-filtered-epochsN.fif"
epochs_n.save(epochs_fname,overwrite=True)
del epochs_n
metadata_nminus_fname = my_path + f"/Segments/session{session}_sub{subject}-filtered-epochsNminus.csv"
pd.to_csv(metadata_nminus_fname)
del metadata_nminus
metadata_nplus_fname = my_path + f"/Segments/session{session}_sub{subject}-filtered-epochsNplus.csv"
pd.to_csv(metadata_nplus_fname)
del metadata_nplus

#epochs = get_wordfreq(epochs) #get word frequency metadata
#epochs = get_word2vec(epochs) #get word2vec vectors metadata

#dsm = Model_DSM(epochs_n,var='w2v')
#data_dsm = generate_meg_dsms(epochs_n)