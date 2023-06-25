# RSA to words n, n+1 and n-1

import mne
import pandas as pd
import numpy as np
import nltk
from wordfreq import zipf_frequency
import mne_rsa
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


def get_consecutive_contents(epochs):
    '''get epochs with three content words in a row'''
    word_category_array = np.array([epoch.metadata['word_category'] for epoch in epochs])
    mask = np.logical_and(word_category_array[:-2] == 'Content',
                        word_category_array[1:-1] == 'Content',
                        word_category_array[2:] == 'Content')
    filtered_epochs = [epoch for epoch, is_filtered in zip(epochs, mask) if not is_filtered]
    return filtered_epochs

    
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


# correlate model and data RDMs
def MEG_DSM_onevar(dsm, data_dsm, operation=str):
    if operation not in ['rsa_n', 'rsa_nplus', 'rsa_nminus']:
        raise ValueError("Invalid operation. Accepted operations are: 'rsa_n', 'rsa_nplus', 'rsa_nminus'")
    rsa = None
    n_times = data_dsm.shape[1]
    # processing of N
    if operation == 'rsa_n':
        data_dsm_modified = [data_dsm[i] for i in range(data_dsm.shape[0])]
        x2 = dsm
        rsa = mne_rsa.rsa(data_dsm_modified, x2, metric='spearman',
                                verbose=True, n_data_dsms=n_times, n_jobs=1)
    # prediction of N+1
    if operation == 'rsa_nplus':
        x1 = data_dsm[:,:data_dsm.shape[1]-1] #drop last item
        data_dsm_dropend = [x1[i] for i in range(x1.shape[0])]
        x2 = dsm[1:] #drop first itme
        rsa = mne_rsa.rsa(data_dsm_dropend, x2, metric='spearman',
                                verbose=True, n_data_dsms=n_times, n_jobs=1)  
    # maintainance of N-1
    if operation == 'rsa_nminus':
        x1 = data_dsm[:,1:data_dsm.shape[1]] #drop first item
        data_dsm_dropend = [x1[i] for i in range(x1.shape[0])]
        x2 = dsm[:dsm.shape[0]-1] #drop last itme
        rsa = mne_rsa.rsa(data_dsm_dropend, x2, metric='spearman',
                                verbose=True, n_data_dsms=n_times, n_jobs=1)
        
    return rsa


#########################################################################
## Get MEG data and corresponding words
my_path = my_path = r'\\rstore.uit.tufts.edu\as_rsch_NCL02$\USERS\Lin\MASC-MEG'
subject='01'
epochs_fname = my_path + f"/Segments/sub{subject}"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction
epochs = get_consecutive_contents(epochs) #filter epochs: only keep trials with three consecutive content words
epochs = get_wordfreq(epochs) #get word frequency metadata
epochs = get_word2vec(epochs) #get word2vec vectors metadata

dsm = Model_DSM(epochs,var='w2v')
data_dsm = generate_meg_dsms(epochs)

'''np.random.seed(42)
random_indices = np.random.choice(dsm.shape[0], size=5000, replace=False)
selected_dsm = dsm[random_indices]
selected_data_dsm = data_dsm[:, random_indices]
'''
selected_dsm = dsm
selected_data_dsm = data_dsm
rsa_n = MEG_DSM_onevar(selected_dsm, selected_data_dsm, 'rsa_n')
rsa_nplus = MEG_DSM_onevar(selected_dsm, selected_data_dsm, 'rsa_nplus')
rsa_nminus = MEG_DSM_onevar(selected_dsm, selected_data_dsm, 'rsa_nminus')

## Plot the RSA values to word2vec representations over time
plt.figure(figsize=(8, 4))
plt.plot(epochs.times, rsa_n)
plt.plot(epochs.times, rsa_nplus)
plt.plot(epochs.times, rsa_nminus)
plt.xlabel('time (s)')
plt.ylabel('RSA value')
plt.legend(['n','n+1', 'n-1'])
plt.savefig(my_path + f"/Segments/sub{subject}"+'_rsa_timecourse')

#########################################################################
# see other examples: https://github.com/wmvanvliet/mne-rsa/tree/master
