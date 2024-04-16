# get epochs for consecutive content trials: words n, n+1 and n-1
import json
import pickle
import mne
import pandas as pd
import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test
import os
from wordfreq import zipf_frequency
import mne_rsa
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy import stats
import gensim.downloader as api
model = api.load("word2vec-google-news-300")

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
    metadata['word_freq'] = metadata['words'].apply(wfreq)
    return metadata


def get_word2vec(metadata):
    '''get word2vec representation for each word'''
    word_list = metadata['words'].tolist()
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
    if len(metadata[var].values[0].shape) == 0: #for lexical vars
        lex_values = metadata[var].values
        diff_matrix = np.abs(np.subtract.outer(lex_values, lex_values))
        indices = np.triu_indices(len(lex_values), k=1)
        higher_diagonal = diff_matrix[indices]
        dsm = higher_diagonal.flatten()
    elif len(metadata[var].values[0].shape) == 1: #for vectors
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
def MEG_DSM_onevar(model_dsm, data_dsm):
    '''model_dsm: array of trialpair
    data_dsm: array of time*trialpair'''
    n_times = data_dsm.shape[0]
    data_dsm_modified = [data_dsm[i] for i in range(data_dsm.shape[0])]
    rsa = mne_rsa.rsa(data_dsm_modified, model_dsm, metric='spearman',
                            verbose=True, n_data_dsms=n_times, n_jobs=1)
    return rsa


# ----------------------------------------------------------------
# run RSA: with different conditions; all trials
# ----------------------------------------------------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]

# get unique subject IDs
subIDs = list(set([file[:5] for file in file_lists]))

# Define Functions for processing data of the same subject
def process_subject(sub, file_lists, my_path):
    sub_files = [file for file in file_lists if file.startswith(sub)]
    
    current_sub = []
    predicted_sub = []
    previous_sub = []
    shuffled_sub = []
    
    for sub_file in sub_files:
        print(f'processing file: {sub_file}')   
        epochs_fname = f"{my_path}/segments/{sub_file}"
        epochs = mne.read_epochs(epochs_fname)
        epochs.apply_baseline((-0.2, 0))
        epochs = epochs[5:]
        
        x = epochs[epochs.metadata['probs'] > 0.1] #only keep trials with probs > 0.1
             
        model_metadata = get_word2vec(epochs.metadata) #epoch 1 --> item 1; including all trials
        w2v_model_rdm = Model_DSM(model_metadata, var='w2v')
        meg_rdm = generate_meg_dsms(epochs)
        rsa_current = MEG_DSM_onevar(w2v_model_rdm, meg_rdm)
        
        model_metadata_forward = model_metadata.shift(-1).iloc[:-1] #epoch 1 --> item 2; remove the last trial
        w2v_model_rdm_forward = Model_DSM(model_metadata_forward, var='w2v')
        meg_rdm_removelast = generate_meg_dsms(epochs[:-1])
        rsa_predicted = MEG_DSM_onevar(w2v_model_rdm_forward, meg_rdm_removelast)
        
        model_metadata_backward = model_metadata.shift(1).iloc[1:] #epoch 2 --> item 1; remove the first trial
        w2v_model_rdm_backward = Model_DSM(model_metadata_backward, var='w2v')
        meg_rdm_removefirst = generate_meg_dsms(epochs[1:])
        rsa_previous = MEG_DSM_onevar(w2v_model_rdm_backward, meg_rdm_removefirst)
        
        shuffled_metadata = model_metadata.sample(frac=1) #randomize all trials
        w2v_model_rdm_shuffled = Model_DSM(shuffled_metadata, var='w2v')
        rsa_shuffled = MEG_DSM_onevar(w2v_model_rdm_shuffled, meg_rdm)
        
        current_sub.append(rsa_current)
        predicted_sub.append(rsa_predicted)
        previous_sub.append(rsa_previous)
        shuffled_sub.append(rsa_shuffled)
    
    rsa_current_mean = np.mean(np.vstack(current_sub), 0)
    rsa_predicted_mean = np.mean(np.vstack(predicted_sub), 0)
    rsa_previous_mean = np.mean(np.vstack(previous_sub), 0)
    rsa_shuffled_mean = np.mean(np.vstack(shuffled_sub), 0)
    
    return {
        'rsa_current': rsa_current_mean,
        'rsa_predicted': rsa_predicted_mean,
        'rsa_previous': rsa_previous_mean,
        'rsa_shuffled': rsa_shuffled_mean
    }


# Process each subject and store results
rsa_results = {
    'subject_id': [],
    'rsa_current': [],
    'rsa_predicted': [],
    'rsa_previous': [],
    'rsa_shuffled': []
}

for sub in subIDs:
    print(f'processing sub {sub}')
    sub_result = process_subject(sub, file_lists, my_path)
    rsa_results['subject_id'].append(sub)
    rsa_results['rsa_current'].append(sub_result['rsa_current'])
    rsa_results['rsa_predicted'].append(sub_result['rsa_predicted'])
    rsa_results['rsa_previous'].append(sub_result['rsa_previous'])
    rsa_results['rsa_shuffled'].append(sub_result['rsa_shuffled'])

# Convert the nested dictionary to a pandas DataFrame
rsa_df = pd.DataFrame(rsa_results)
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec_allCloze.json'
rsa_df.to_json(fname)

print('Done')

# ----------------------------------------------------------------
# Statistical tests: MNE-python function, cluster-based permutation
# ----------------------------------------------------------------
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec_allCloze.json'
df = pd.read_json(fname)
data_shuffled = np.vstack(df['rsa_shuffled'].to_list())

timewind = np.linspace(-0.2, 1.0, 121) * 1000

allconds = ['rsa_current', 'rsa_predicted', 'rsa_previous']
stat_output={}
for cond in allconds:
    data_obs = np.vstack(df[cond].to_list())
    dif = data_obs - data_shuffled
    fdata = np.reshape(dif,(dif.shape[0],dif.shape[1],1))

    # Run the cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(fdata, n_permutations=1000)

    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values <= 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]

    sig_pvals = cluster_p_values[good_clusters_idx]
    output = {}
    for sig_n in range(len(sig_pvals)):
        output[f'pval_cluster{sig_n}'] = sig_pvals[sig_n]
        output[f'startT_cluster{sig_n}'] = round(timewind[good_clusters[sig_n][0][0]])
        output[f'endT_cluster{sig_n}'] = round(timewind[good_clusters[sig_n][0][-1]])
    stat_output[cond] = output

fname = 'S:/USERS/Lin/MASC-MEG/RSA/word2vec_allCloze_stats.pkl'
with open(fname,'wb') as pkl_file:
    pickle.dump(stat_output,pkl_file)


#------------------------------
# plot the three RSA effects
#------------------------------
def plot_R(data, color, condlabel, sig_time_winds,plot_bars=True):
    '''data: subject x time'''      
    meanR = np.mean(data, 0)
    sd = np.std(data, 0) / np.sqrt(data.shape[0])
    timewind = np.linspace(-0.2, 1.0, meanR.shape[0]) * 1000
    plt.plot(timewind, meanR, color=color, label=condlabel)
    plt.fill_between(timewind,
                    meanR - sd,
                    meanR + sd,
                    color=color,
                    alpha=0.2)
    if plot_bars:
        # Calculate y-coordinate for the horizontal bars
        min_meanR_sd = np.min(meanR - sd)
        bar_height = 0.1 * np.abs(min_meanR_sd)
        # Add horizontal bars for significant time ranges
        for cluster, time_range in sig_time_winds.items():
            if cluster.startswith('startT_cluster'):
                cluster_num = cluster.split('_')[-1] # Extract cluster number
                start_time = time_range
                end_time = sig_time_winds[f'endT_{cluster_num}']
                plt.axhspan(min_meanR_sd - 0.5 * bar_height, min_meanR_sd - 0.5 * bar_height + bar_height,
                            xmin=(start_time - timewind[0]) / (timewind[-1] - timewind[0]),
                            xmax=(end_time - timewind[0]) / (timewind[-1] - timewind[0]),
                            color=color, alpha=0.3)

plot_R(np.vstack(df['rsa_current'].to_list()),'red','current N',stat_output['rsa_current'],plot_bars=True) #current
plot_R(np.vstack(df['rsa_previous'].to_list()),'green','previous Nminus',stat_output['rsa_previous'],plot_bars=True) #previous
plot_R(np.vstack(df['rsa_predicted'].to_list()),'blue','predicted Nplus',stat_output['rsa_predicted'],plot_bars=True) #predicted

plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('R')
plt.title('RSA')
plt.legend()
plt.show()


plt.plot(timewind,np.mean(np.vstack(df['rsa_current'].to_numpy()),0),'r')
plt.plot(timewind,np.mean(np.vstack(df['rsa_previous'].to_numpy()),0),'g')
plt.plot(timewind,np.mean(np.vstack(df['rsa_predicted'].to_numpy()),0),'b')
plt.plot(timewind,np.mean(np.vstack(df['rsa_shuffled'].to_numpy()),0),'k')


# ----------------------------------------------------------------
# Plot results: line plots
# ----------------------------------------------------------------
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec.json'
df=pd.read_json(fname)
timewind = np.linspace(-0.2,1.0,121)
plt.plot(timewind,np.mean(np.vstack(df['lowCloze_preMEG_preW2V'].to_numpy()),0),'r',label='lowCloze_preMEG_preW2V')
plt.plot(timewind,np.mean(np.vstack(df['midCloze_preMEG_preW2V'].to_numpy()),0),'g',label='midCloze_preMEG_preW2V')
plt.plot(timewind,np.mean(np.vstack(df['highCloze_preMEG_preW2V'].to_numpy()),0),'b',label='highCloze_preMEG_preW2V')
plt.legend()
plt.show()

plt.plot(timewind,np.mean(np.vstack(df['lowCloze_preMEG_cwW2V'].to_numpy()),0),'r')
plt.plot(timewind,np.mean(np.vstack(df['midCloze_preMEG_cwW2V'].to_numpy()),0),'g')
plt.plot(timewind,np.mean(np.vstack(df['highCloze_preMEG_cwW2V'].to_numpy()),0),'b')
plt.show()

plt.plot(timewind,np.mean(np.vstack(df['lowCloze_cwMEG_cwW2V'].to_numpy()),0),'r')
plt.plot(timewind,np.mean(np.vstack(df['midCloze_cwMEG_cwW2V'].to_numpy()),0),'g')
plt.plot(timewind,np.mean(np.vstack(df['highCloze_cwMEG_cwW2V'].to_numpy()),0),'b')
plt.show()

plt.plot(timewind,np.mean(np.vstack(df['lowCloze_cwMEG_preW2V'].to_numpy()),0),'r')
plt.plot(timewind,np.mean(np.vstack(df['midCloze_cwMEG_preW2V'].to_numpy()),0),'g')
plt.plot(timewind,np.mean(np.vstack(df['highCloze_cwMEG_preW2V'].to_numpy()),0),'b')
plt.show()


fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec.json'
df=pd.read_json(fname)
timewind = np.linspace(-0.2,1.0,121)
plt.plot(timewind,np.mean(np.vstack(df['allCloze_preMEG_preW2V'].to_numpy()),0),'m',label='allCloze_preMEG_preW2V')
plt.plot(timewind,np.mean(np.vstack(df['allCloze_preMEG_cwW2V'].to_numpy()),0),'b',label='allCloze_preMEG_cwW2V')
plt.plot(timewind,np.mean(np.vstack(df['allCloze_cwMEG_cwW2V'].to_numpy()),0),'k',label='allCloze_cwMEG_cwW2V')
plt.plot(timewind,np.mean(np.vstack(df['allCloze_cwMEG_preW2V'].to_numpy()),0),'r',label='allCloze_cwMEG_preW2V')
plt.show()

# ----------------------------------------------------------------
# Statistical test at each time point independently
# ----------------------------------------------------------------
allconds = ['allCloze_preMEG_preW2V','allCloze_preMEG_cwW2V','allCloze_cwMEG_preW2V','allCloze_cwMEG_cwW2V']
null_hypothesis_mean = 0
tval = {}
pval = {}
for cond in allconds:
    data = cond_rsa[cond]
    t_statistic, p_value = stats.ttest_1samp(data, null_hypothesis_mean)
    tval[cond]=t_statistic
    pval[cond]=p_value
data = cond_rsa['allCloze_cwMEG_preW2V']
null_hypothesis_mean = 0
t_statistic, p_value = stats.ttest_1samp(data, null_hypothesis_mean)


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
            plot_n = np.mean(np.array(rsa_n),0)
            plot_nplus = np.mean(np.array(rsa_nplus),0)
            plot_nminus = np.mean(np.array(rsa_nminus),0)
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