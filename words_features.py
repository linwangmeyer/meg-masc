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

'''
def get_consecutive_contents(epochs):
    #'''#get epochs with three content words in a row'''
    filtered_epochs_n = []
    filtered_epochs_nminus = []
    filtered_epochs_nplus = []
    for i in tqdm(range(1, len(epochs) - 1)):
        cat_i = epochs[i].metadata['word_category']
        cat_i_minus = epochs[i-1].metadata['word_category']
        cat_i_plus = epochs[i+1].metadata['word_category']
        if cat_i.item() == 'Content' and cat_i_minus.item() == 'Content' and cat_i_plus.item() == 'Content':
            filtered_epochs_n.append(epochs[i])            
            filtered_epochs_nminus.append(epochs[i-1])
            filtered_epochs_nplus.append(epochs[i+1])
    epochs_n = mne.concatenate_epochs(filtered_epochs_n)
    epochs_nminus = mne.concatenate_epochs(filtered_epochs_nminus)
    epochs_nplus = mne.concatenate_epochs(filtered_epochs_nplus)
    return epochs_n,epochs_nminus,epochs_nplus
'''

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
    if len(metadata[var].values[0].shape) == 0: #if the values are single numbers, e.g. length
        num_letters = metadata[var].values
        diff_matrix = np.abs(np.subtract.outer(num_letters, num_letters))
        indices = np.triu_indices(len(num_letters), k=1)
        higher_diagonal = diff_matrix[indices]
        dsm = higher_diagonal.flatten()
    elif len(metadata[var].values[0].shape) == 1: # if values are vectors, e.g. word2vec
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
def MEG_DSM_onevar(dsm, data_dsm):
    rsa = None
    n_times = data_dsm.shape[1]    
    data_dsm_modified = [data_dsm[i] for i in range(data_dsm.shape[0])]
    x2 = dsm
    rsa = mne_rsa.rsa(data_dsm_modified, x2, metric='spearman',
                            verbose=True, n_data_dsms=n_times, n_jobs=1)
    return rsa


'''
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
'''


#########################################################################
## Get MEG data and corresponding words
my_path = r'\\rstore.uit.tufts.edu\as_rsch_NCL02$\USERS\Lin\MASC-MEG'
subjects = range(1,12)
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(1):
        filtered_epochs = {}
        epochs_fname = my_path + f"/Segments/session{session}_sub{subject}"
        epochs = mne.read_epochs(epochs_fname)
        epochs.drop_channels(epochs.info['bads']) #drop bad channels
        epochs.apply_baseline() #baseline correction
        epochs_n,epochs_nminus,epochs_nplus = get_consecutive_contents(epochs, batch_size=10)
        #epochs_n,epochs_nminus,epochs_nplus = get_consecutive_contents(epochs) #filter epochs: only keep trials with three consecutive content words
        
        # get more metadata
        epochs_n = get_wordfreq(epochs_n) #get word frequency metadata
        epochs_n = get_word2vec(epochs_n) #get word2vec vectors metadata
        
        epochs_nminus = get_wordfreq(epochs_nminus) #get word frequency metadata
        epochs_nminus = get_word2vec(epochs_nminus) #get word2vec vectors metadata
        
        epochs_nplus = get_wordfreq(epochs_nplus) #get word frequency metadata
        epochs_nplus = get_word2vec(epochs_nplus) #get word2vec vectors metadata
        
        filtered_epochs['n'] = epochs_n
        filtered_epochs['n-1'] = epochs_nminus
        filtered_epochs['n+1'] = epochs_nplus
        epochs_fname = my_path + f"/RSA/session{session}_sub{subject}"
        filtered_epochs.save(epochs_fname,overwrite=True)

# Run RSA
subjects = range(1,12)
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(1):
        rsa_all = {}
        epochs_fname = my_path + f"/RSA/session{session}_sub{subject}"
        filtered_epochs = mne.read_epochs(epochs_fname)

        epochs_n = filtered_epochs['n']
        epochs_nminus = filtered_epochs['n-1']
        epochs_nplus = filtered_epochs['n+1']
        
        # build dsm
        dsm_n = Model_DSM(epochs_n,var='w2v')
        dsm_nminus = Model_DSM(epochs_nminus,var='w2v')
        dsm_nplus = Model_DSM(epochs_nplus,var='w2v')
        data_dsm = generate_meg_dsms(epochs_n)

        # run RSA
        rsa_n = MEG_DSM_onevar(dsm_n, data_dsm)
        rsa_nplus = MEG_DSM_onevar(dsm_nminus, data_dsm)
        rsa_nminus = MEG_DSM_onevar(dsm_nplus, data_dsm)

        rsa_all['n'] = rsa_n
        rsa_all['n-1'] = rsa_nminus
        rsa_all['n+1'] = rsa_nplus
        
        # Save the dictionary to a file
        rsa_fname = my_path + f"/RSA/session{session}_sub{subject}"+'_rsa_timecourse'
        with open(rsa_fname, "w") as file:
            json.dump(rsa_all, file)
        
        ## Plot the RSA values to word2vec representations over time
        plt.figure(figsize=(8, 4))
        plt.plot(epochs.times, rsa_n)
        plt.plot(epochs.times, rsa_nplus)
        plt.plot(epochs.times, rsa_nminus)
        plt.xlabel('time (s)')
        plt.ylabel('RSA value')
        plt.legend(['n','n+1', 'n-1'])
        plt.savefig(my_path + f"/RSA/figures/session{session}_sub{subject}"+'_rsa_timecourse')
        plt.close()


subjects = range(1,12)
for i in subjects:
    subject = str(i).zfill(2)
    rsa_subs = []
    for session in range(1):
        rsa_fname = my_path + f"/RSA/session{session}_sub{subject}"+'_rsa_timecourse'
        with open(rsa_fname, "r") as file:
            rsa_all = json.load(file)
        rsa_n = rsa_all['n']
        rsa_nminus = rsa_all['n-1']
        rsa_nplus = rsa_all['n+1']
        
        
#########################################################################
# see other examples: https://github.com/wmvanvliet/mne-rsa/tree/master
# https://users.aalto.fi/~vanvlm1/mne-rsa/index.html
