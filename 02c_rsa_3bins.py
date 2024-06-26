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


def calculate_all_rdms_cw(epochs, condition_range):
    '''calculate the RDMs for both MEG data and lexico-semantic variables for cw
    epochs: mne-segmented epochs
    condition_range: list of tuples to define the cloze ranges,
    e.g. [(0.05, 0.20, 'low'), (0.20, 0.60, 'mid'), (0.60, 1.0, 'high')]
    return: dictionary containing all RDMs'''
    all_rdms = {}
    metadata = epochs.metadata
    for condition in condition_range:
        x=(metadata['probs'] > condition[0]) & (metadata['probs'] <= condition[1])
        sel_epochs = epochs[1:][x[1:]]#remove the first epoch (because there is no preceding trial for the first epoch)
        meg_rdm = generate_meg_dsms(sel_epochs)
        model_metadata = get_word2vec(sel_epochs.metadata)
        w2v_model_rdm = Model_DSM(model_metadata, var='w2v')
        freq_model_rdm = Model_DSM(model_metadata, var='frequency')
        probs_model_rdm = Model_DSM(model_metadata, var='probs')
        duration_model_rdm = Model_DSM(model_metadata, var='duration')

        all_rdms[f'meg_{condition[2]}'] = meg_rdm
        all_rdms[f'w2v_{condition[2]}'] = w2v_model_rdm
        all_rdms[f'freq_{condition[2]}'] = freq_model_rdm
        all_rdms[f'probs_{condition[2]}'] = probs_model_rdm
        all_rdms[f'duration_{condition[2]}'] = duration_model_rdm
    return all_rdms

def calculate_all_rdms_precw(epochs, condition_range):
    '''calculate the RDMs for both MEG data and lexico-semantic variables for words preceding cw (precw)
    epochs: mne-segmented epochs
    condition_range: list of tuples to define the cloze ranges,
    e.g. [(0.05, 0.20, 'low'), (0.20, 0.60, 'mid'), (0.60, 1.0, 'high')]
    return: dictionary containing all RDMs'''
    all_rdms = {}
    metadata = epochs.metadata
    for condition in condition_range:
        x=(metadata['probs'] > condition[0]) & (metadata['probs'] <= condition[1])
        x_shift=x.copy().shift(-1) #selecting the immediate preceding epoch
        epochs_shift = epochs[x_shift.fillna(False)]#remove the last epoch by setting it to False
        meg_rdm = generate_meg_dsms(epochs_shift)
        model_metadata = get_word2vec(epochs_shift.metadata)
        w2v_model_rdm = Model_DSM(model_metadata, var='w2v')
        freq_model_rdm = Model_DSM(model_metadata, var='frequency')
        probs_model_rdm = Model_DSM(model_metadata, var='probs')
        duration_model_rdm = Model_DSM(model_metadata, var='duration')

        all_rdms[f'meg_{condition[2]}'] = meg_rdm
        all_rdms[f'w2v_{condition[2]}'] = w2v_model_rdm
        all_rdms[f'freq_{condition[2]}'] = freq_model_rdm
        all_rdms[f'probs_{condition[2]}'] = probs_model_rdm
        all_rdms[f'duration_{condition[2]}'] = duration_model_rdm
    return all_rdms



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
# run RSA: corresponding orders
# ----------------------------------------------------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
rsa_grand=[]
for file in file_lists:
    
    print(f'processing file: {file}')   
        
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    epochs.apply_baseline((-0.2, 0))
    epochs = epochs[5:] #remove the first 5 trials

    all_words = epochs.metadata['words']
    missing_words = set([word for word in all_words if word not in model.key_to_index])
    if len(missing_words)>0:
        raise ValueError (f'words missing in word2vec in {file}')
    
    # Define conditions
    condition_ranges = [(0, 1.0, 'all'), (0.05, 0.20, 'low'), (0.20, 0.60, 'mid'), (0.60, 1.0, 'high')]
    all_rdms_precw = calculate_all_rdms_precw(epochs, condition_ranges)
    all_rdms_cw = calculate_all_rdms_cw(epochs, condition_ranges)

    # RSA
    cloze_list = ['all','low','mid','high']
    rsa_all = {}
    for cloze_cond in cloze_list:
        meg_precw = all_rdms_precw['meg_'+cloze_cond]
        meg_cw = all_rdms_cw['meg_'+cloze_cond]
        
        w2v_precw = all_rdms_precw['w2v_'+cloze_cond]
        w2v_cw = all_rdms_cw['w2v_'+cloze_cond]
        
        rsa_all[cloze_cond+'Cloze' + '_preMEG' + '_preW2V'] = MEG_DSM_onevar(w2v_precw,meg_precw)
        rsa_all[cloze_cond+'Cloze' + '_preMEG' + '_cwW2V'] = MEG_DSM_onevar(w2v_cw,meg_precw)
        rsa_all[cloze_cond+'Cloze' + '_cwMEG' + '_preW2V'] = MEG_DSM_onevar(w2v_precw,meg_cw)
        rsa_all[cloze_cond+'Cloze' + '_cwMEG' + '_cwW2V'] = MEG_DSM_onevar(w2v_cw,meg_cw)
    
    rsa_grand.append(rsa_all)

# get all data together
df = pd.DataFrame(rsa_grand[:])
df['ID'] = [file[:-4] for file in file_lists]
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec.json'
df.to_json(fname)


# ----------------------------------------------------------------
# run RSA: permuted epochs
# ----------------------------------------------------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]

# get unique subject IDs
subIDs = list(set([file[:5] for file in file_lists]))
rsa_grand=[]
for sub in subIDs:
    sub_files = [file for file in file_lists if file.startswith(sub)]
    
    rsa_sub = []
    for sub_file in sub_files:
    
        print(f'processing file: {sub_file}')   
            
        # get clean epochs of experimental conditions
        epochs_fname = my_path + f"/segments/{file}"
        epochs = mne.read_epochs(epochs_fname)
        epochs.apply_baseline((-0.2, 0))
        epochs = epochs[5:] #remove the first 5 trials
        sel_epochs = epochs[1:]#remove the first epoch (because there is no preceding trial for the first epoch)
        
        # get the RDM for MEG data
        meg_rdm = generate_meg_dsms(sel_epochs)
        
        # get the word2vec embeddings of unshuffled data
        model_metadata = get_word2vec(sel_epochs.metadata)
        
        # Shuffle the metadata DataFrame, and the word2vec similarity for the shuffled data
        shuffled_metadata = model_metadata.sample(frac=1).reset_index(drop=True)
        w2v_model_rdm = Model_DSM(shuffled_metadata, var='w2v')
        perm_rsa = MEG_DSM_onevar(w2v_model_rdm,meg_rdm)
        rsa_sub.append(perm_rsa)
    #average across sessions and tasks
    rsa_sub_mean = np.mean(np.vstack(rsa_sub),0)
    rsa_grand.append(rsa_sub_mean)

df = pd.DataFrame(rsa_grand[:])
df['ID'] = subIDs
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec_shuffled.json'
df.to_json(fname)


# ----------------------------------------------------------------
# Average results for each subject
# Combine results with shuffled data
# ----------------------------------------------------------------
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec.json'
df = pd.read_json(fname)
subIDs = set(df['ID'].str.split('_').str[0])
cond_rsa = {}
allconds = ['allCloze_preMEG_cwW2V','allCloze_cwMEG_preW2V','allCloze_cwMEG_cwW2V',
            'highCloze_preMEG_cwW2V','midCloze_preMEG_cwW2V','lowCloze_preMEG_cwW2V',
            'highCloze_cwMEG_preW2V','midCloze_cwMEG_preW2V','lowCloze_cwMEG_preW2V',
            'highCloze_cwMEG_cwW2V','midCloze_cwMEG_cwW2V','lowCloze_cwMEG_cwW2V']
for cond in allconds:
    all_rsa = []
    for sub in subIDs:    
        df2 = df[df['ID'].str.startswith(sub)]
        rsa = np.array(df2[cond].tolist()).mean(axis=0)
        all_rsa.append(rsa)
    cond_rsa[cond] = np.vstack(all_rsa[:])

# get the shuffled RSA results
fname='S:/USERS/Lin/MASC-MEG/RSA/word2vec_shuffled.json'
df_shuffle = pd.read_json(fname)
data_shuffle = df_shuffle.iloc[:,:121].values
cond_rsa['shuffle'] = data_shuffle

# Saving dictionary to a Pickle file
fname_subs='S:/USERS/Lin/MASC-MEG/RSA/word2vec_all.pkl'
with open(fname_subs, 'wb') as pickle_file:
    pickle.dump(cond_rsa, pickle_file)
    
# ----------------------------------------------------------------
# Statistical test: MNE-python function, cluster-based permutation
# ----------------------------------------------------------------
# get RSA results of all conditions
fname_subs='S:/USERS/Lin/MASC-MEG/RSA/word2vec_all.pkl'
with open(fname_subs, 'rb') as pickle_file:
    cond_rsa = pickle.load(pickle_file)

allconds = ['shuffle','allCloze_preMEG_cwW2V', 'allCloze_cwMEG_preW2V', 'allCloze_cwMEG_cwW2V',
            'highCloze_preMEG_cwW2V','midCloze_preMEG_cwW2V','lowCloze_preMEG_cwW2V',
            'highCloze_cwMEG_preW2V','midCloze_cwMEG_preW2V','lowCloze_cwMEG_preW2V',
            'highCloze_cwMEG_cwW2V','midCloze_cwMEG_cwW2V','lowCloze_cwMEG_cwW2V']
data_shuffle = cond_rsa['shuffle']
timewind = np.linspace(-0.2, 1.0, 121) * 1000
stat_output={}
for cond in allconds[1:]:
    data_obs = cond_rsa[cond]
    dif = data_obs - data_shuffle
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
fname = 'S:/USERS/Lin/MASC-MEG/RSA/word2vec_stats.pkl'
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
                            color=color, alpha=0.3, label=f'{cluster_num}')
    
plot_R(cond_rsa['allCloze_cwMEG_cwW2V'],'red','current N',stat_output['allCloze_cwMEG_cwW2V'],plot_bars=True) #current
plot_R(cond_rsa['allCloze_cwMEG_preW2V'],'green','previous Nminus',stat_output['allCloze_cwMEG_preW2V'],plot_bars=True) #previous
plot_R(cond_rsa['allCloze_preMEG_cwW2V'],'blue','predicted Nplus',stat_output['allCloze_preMEG_cwW2V'],plot_bars=True) #predicted

plot_R(cond_rsa['highCloze_preMEG_cwW2V'],'blue','predicted Nplus: hc',stat_output['highCloze_preMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['midCloze_preMEG_cwW2V'],'deepskyblue','predicted Nplus: md',stat_output['midCloze_preMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['lowCloze_preMEG_cwW2V'],'dodgerblue','predicted Nplus: lc',stat_output['lowCloze_preMEG_cwW2V'],plot_bars=True) #predicted

plot_R(cond_rsa['highCloze_cwMEG_cwW2V'],'red','current N: hc',stat_output['highCloze_cwMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['midCloze_cwMEG_cwW2V'],'indianred','current N: mc',stat_output['midCloze_cwMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['lowCloze_cwMEG_cwW2V'],'darkred','current N: lc',stat_output['lowCloze_cwMEG_cwW2V'],plot_bars=True) #predicted

plot_R(cond_rsa['highCloze_cwMEG_cwW2V'],'green','previous Nminus: hc',stat_output['highCloze_cwMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['midCloze_cwMEG_cwW2V'],'lime','previous Nminus: mc',stat_output['midCloze_cwMEG_cwW2V'],plot_bars=True) #predicted
plot_R(cond_rsa['lowCloze_cwMEG_cwW2V'],'forestgreen','previous Nminus: lc',stat_output['lowCloze_cwMEG_cwW2V'],plot_bars=True) #predicted

plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('R')
plt.title('RSA')
plt.legend()
plt.show()

plt.plot(np.mean(cond_rsa['highCloze_preMEG_cwW2V'],0),'b')
plt.plot(np.mean(cond_rsa['allCloze_preMEG_cwW2V'],0),'k')

plt.plot(timewind,np.mean(np.vstack(df['highCloze_preMEG_cwW2V'].to_numpy()),0),'b')
plt.plot(timewind,np.mean(np.vstack(df['allCloze_preMEG_cwW2V'].to_numpy()),0),'k')
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