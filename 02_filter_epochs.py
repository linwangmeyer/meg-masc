# get epochs for consecutive content trials: words n, n+1 and n-1
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
    return metadata


def Model_DSM(metadata,var=str):
    '''get pairwise similarity for a variable of interest'''
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
def MEG_DSM_onevar(dsm, data_dsm):
    n_times = data_dsm.shape[1]
    data_dsm_modified = [data_dsm[i] for i in range(data_dsm.shape[0])]
    x2 = dsm
    rsa = mne_rsa.rsa(data_dsm_modified, x2, metric='spearman',
                            verbose=True, n_data_dsms=n_times, n_jobs=1)
    
    return rsa


################################################################
## Run functions: filter epochs
################################################################
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
#my_path = r'/cluster/home/lwang11/MASC-MEG/'
subject = str(1).zfill(2)
session = 0
task = 0
epochs_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}"
epochs = mne.read_epochs(epochs_fname)
epochs.drop_channels(epochs.info['bads']) #drop bad channels
epochs.apply_baseline() #baseline correction

epochs_n, metadata_nminus, metadata_nplus = get_consecutive_contents(epochs,batch_size=50) #filter epochs: only keep trials with three consecutive content words
epochs_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsN.fif"
epochs_n.save(epochs_fname,overwrite=True)
del epochs_n
metadata_nminus_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsNminus.csv"
metadata_nminus.to_csv(metadata_nminus_fname, index=False)
del metadata_nminus
metadata_nplus_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsNplus.csv"
metadata_nplus.to_csv(metadata_nplus_fname, index=False)
del metadata_nplus

################################################################
# get word features: w2v
epochs_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsN.fif"
epochs_n = mne.read_epochs(epochs_fname)

metadata_nminus_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsNminus.csv"
metadata_nminus = pd.read_csv(metadata_nminus_fname)

metadata_nplus_fname = my_path + f"/segments/session{session}_sub{subject}_task{task}-filtered-epochsNplus.csv"
metadata_nplus = pd.read_csv(metadata_nplus_fname)

# get word frequency
metadata_n = get_wordfreq(epochs_n.metadata)
metadata_nminus = get_wordfreq(metadata_nminus)
metadata_nplus = get_wordfreq(metadata_nplus)

# get word2vec embeddings
metadata_n = get_word2vec(metadata_n)
metadata_nminus = get_word2vec(metadata_nminus)
metadata_nplus = get_word2vec(metadata_nplus)

# get model dsms
w2v_n = Model_DSM(metadata_n,var='w2v')
w2v_nminus = Model_DSM(metadata_nminus,var='w2v')
w2v_nplus = Model_DSM(metadata_nplus,var='w2v')

# get data dsm
data_dsm = generate_meg_dsms(epochs_n)

################################################
# run RSA
rsa_n = MEG_DSM_onevar(w2v_n, data_dsm)
rsa_nplus = MEG_DSM_onevar(w2v_nminus, data_dsm)
rsa_nminus = MEG_DSM_onevar(w2v_nplus, data_dsm)

rsa_all = {}
rsa_all['n'] = rsa_n.tolist()
rsa_all['n-1'] = rsa_nminus.tolist()
rsa_all['n+1'] = rsa_nplus.tolist()

# Save the dictionary to a file
rsa_fname = my_path + f"rsa/session{session}_sub{subject}_task{task}"+'_rsa_timecourse'
with open(rsa_fname, "w") as file:
    json.dump(rsa_all, file)


###############################################################
# plot RSA results for each subject and each session: word2vec
subjects = range(1,12)
for i in subjects:
    subject = str(i).zfill(2)
    rsa_subs = []
    for session in range(1):
        for task in range(4):
        rsa_fname = my_path + f"/rsa/session{session}_sub{subject}_task{task}"+'_rsa_timecourse'
        rsa_n=[]
        rsa_nplus=[]
        rsa_nminus=[]
        with open(rsa_fname, "r") as file:
            rsa_all = json.load(file)         
            rsa_n.append(np.array(rsa_all['n']))
            rsa_nplus.append(np.array(rsa_all['n+1']))
            rsa_nminus.append(np.array(rsa_all['n-1']))
        plot_n = mean(np.array(rsa_n),0)
        plot_nplus = mean(np.array(rsa_nplus),0)
        plot_nminus = mean(np.array(rsa_nminus),0)
        plt.figure(figsize=(8, 4))
        plt.plot(epochs.times, rsa_n)
        plt.plot(epochs.times, rsa_nplus)
        plt.plot(epochs.times, rsa_nminus)
        plt.xlabel('time (s)')
        plt.ylabel('RSA value')
        plt.legend(['n','n+1', 'n-1'])
        plt.savefig(my_path + f"/rsa/figures/session{session}_sub{subject}"+'_rsa_timecourse')
        plt.close()


#########################################################################
# see other examples: https://github.com/wmvanvliet/mne-rsa/tree/master
# https://users.aalto.fi/~vanvlm1/mne-rsa/index.html