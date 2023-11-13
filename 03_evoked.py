# calculate evoked responses
import json
import mne
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from mne.stats.regression import linear_regression_raw
from mne.viz import plot_compare_evokeds
from mne.stats import fdr_correction, linear_regression

matplotlib.use("TkAgg")

################################################################
# evoked activity to different conditions of features: Cloze
################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
for file in file_lists:
    
    print(f'processing file: {file}')   
        
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    
    metadata = epochs.metadata
    epochs_low = epochs[(metadata['probs'] < 0.4) & (metadata['probs'] > 0.1)]
    epochs_high = epochs[metadata['probs'] >= 0.4]

    evokeds_low = epochs_low.average().apply_baseline((-0.2, 0))
    evokeds_high = epochs_high.average().apply_baseline((-0.2, 0))

    evoked_fname = os.path.join(my_path,'ERFs','cloze_'+file)
    mne.write_evokeds(evoked_fname,[evokeds_low, evokeds_high],overwrite=True)

# plot ERFs of individual subject
mne.viz.plot_compare_evokeds([evokeds_low, evokeds_high], picks=['MEG 054'])
mne.viz.plot_compare_evokeds([evokeds_low, evokeds_high], picks=['MEG 065'])


# ------------ Grand average ------------------
file_lists = [file for file in os.listdir(my_path+'ERFs/') if file.endswith(".fif")]
all_low = []
all_high = []
for file in file_lists:
    read_fname = os.path.join(my_path,'ERFs',file)
    print(f'processing file: {file}')
    
    evokeds_low, evokeds_high = mne.read_evokeds(read_fname)    
    all_low.append(evokeds_low)
    all_high.append(evokeds_high)
    
grandavg_low = mne.grand_average(all_low,interpolate_bads=False,drop_bads=True)
grandavg_high = mne.grand_average(all_high,interpolate_bads=False,drop_bads=True)
evoked_fname = os.path.join(my_path,'ERFs','cloze_grandavg.fif')
mne.write_evokeds(evoked_fname,[grandavg_low, grandavg_high],overwrite=True)

evoked = epochs.average()
badsensor = mne.pick_channels(epochs.ch_names,epochs.info['bads'])
picksensor = np.append(badsensor,0)
plt.plot(evoked.get_data()[picksensor,:].T)

evoked_int = evoked.interpolate_bads(reset_bads=True,exclude=[], origin='auto')

##################################################################################
# ------------ Plot Grand average ------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
evoked_fname = os.path.join(my_path,'ERFs','cloze_grandavg.fif')
grandavg = mne.read_evokeds(evoked_fname)
grandavg_low = grandavg[0]
grandavg_high = grandavg[1]

# plot waveforms
mne.viz.plot_compare_evokeds([grandavg_low, grandavg_high], picks=['MEG 134'])
mne.viz.plot_compare_evokeds([grandavg_low, grandavg_high], picks=['MEG 065'])

# plot topograph of the effect
evoked_diff = mne.combine_evoked([grandavg_low, grandavg_high], weights=[1, -1])
evoked_diff.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag")

# check sensor layout
layout = mne.channels.find_layout(epochs.info, ch_type='meg')
layout.plot()




#############################################################################
# evoked activity to different conditions of features: interaction with Cloze
#############################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
all_evokeds = []

for file in file_lists:
    
    print(f'processing file: {file}')   
        
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    epochs.apply_baseline((-0.2, 0))
        
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    
    conditions = [
    ((metadata['probs'] < 0.4) & (metadata['probs'] > 0.1) & (metadata['frequency'] < 5) & (metadata['frequency'] > 1), 'lowProb_lowFreq'),
    ((metadata['probs'] >= 0.4) & (metadata['frequency'] < 5) & (metadata['frequency'] > 1), 'highProb_lowFreq'),
    ((metadata['probs'] < 0.4) & (metadata['probs'] > 0.1) & (metadata['frequency'] >= 5), 'lowProb_highFreq'),
    ((metadata['probs'] >= 0.4) & (metadata['frequency'] >= 5), 'highProb_highFreq'),
    ((metadata['probs'] < 0.4) & (metadata['probs'] > 0.1) & (metadata['length'] < 5), 'lowProb_lowLength'),
    ((metadata['probs'] >= 0.4) & (metadata['length'] < 5), 'highProb_lowLength'),
    ((metadata['probs'] < 0.4) & (metadata['probs'] > 0.1) & (metadata['length'] >= 5), 'lowProb_highLength'),
    ((metadata['probs'] >= 0.4) & (metadata['length'] >= 5), 'highProb_highLength')
    ]

    evokeds = {}
    for condition, label in conditions:
        epochs_condition = epochs[condition]
        evokeds[label] = epochs_condition.average().apply_baseline((-0.2, 0))
    
    all_evokeds.append(evokeds)

# grandaverage
evokeds_by_condition = {}
for evokeds_dict in all_evokeds:
    for condition_name, evoked_obj in evokeds_dict.items():
        if condition_name not in evokeds_by_condition:
            evokeds_by_condition[condition_name] = []
        evokeds_by_condition[condition_name].append(evoked_obj)

grandavg = {}
for cond, evks in evokeds_by_condition.items():
    grandavg[cond] = mne.grand_average(evks,interpolate_bads=False,drop_bads=True)

for condition, evoked in grandavg.items():
    evoked.comment = condition
    evoked_fname = os.path.join(my_path,'ERFs',condition+'.fif')
    mne.write_evokeds(evoked_fname, evoked, overwrite=True)


# --------------------- Plot effects --------------------- #
conditions = ['lowProb_lowFreq',
 'highProb_lowFreq',
 'lowProb_highFreq',
 'highProb_highFreq',
 'lowProb_lowLength',
 'highProb_lowLength',
 'lowProb_highLength',
 'highProb_highLength']
grandavg = {}
for cond in conditions:
    evoked_fname = os.path.join(my_path,'ERFs',cond+'.fif')    
    grandavg[cond] = mne.read_evokeds(evoked_fname)[0]

## Cloze x frequency
plot_sensor = mne.pick_channels(grandavg['highProb_highFreq'].ch_names,['MEG 065'])
highProb_highFreq = grandavg['highProb_highFreq'].get_data()[plot_sensor,:]
lowProb_highFreq = grandavg['lowProb_highFreq'].get_data()[plot_sensor,:]
highProb_lowFreq = grandavg['highProb_lowFreq'].get_data()[plot_sensor,:]
lowProb_lowFreq = grandavg['lowProb_lowFreq'].get_data()[plot_sensor,:]
plt.plot(np.linspace(-0.2, 1.0, 121), highProb_highFreq[0], color='b', label='HighProb_highFreq', linestyle='-')
plt.plot(np.linspace(-0.2, 1.0, 121), lowProb_highFreq[0], color='r', label='LowProb_highFreq', linestyle='-')
plt.plot(np.linspace(-0.2, 1.0, 121), highProb_lowFreq[0], color='b', label='HighProb_lowFreq', linestyle='--')
plt.plot(np.linspace(-0.2, 1.0, 121), lowProb_lowFreq[0], color='r', label='LowProb_lowFreq', linestyle='--')
plt.legend()
plt.savefig(os.path.join(my_path,'ERFs','grandavg_interaction_clozeBYfreq.png'))
plt.show()

## cloze x length
highProb_highLength = grandavg['highProb_highLength'].get_data()[plot_sensor,:]
lowProb_highLength = grandavg['lowProb_highLength'].get_data()[plot_sensor,:]
highProb_lowLength = grandavg['highProb_lowLength'].get_data()[plot_sensor,:]
lowProb_lowLength = grandavg['lowProb_lowLength'].get_data()[plot_sensor,:]
plt.plot(np.linspace(-0.2, 1.0, 121), highProb_highLength[0], color='b', label='HighProb_highLength', linestyle='-')
plt.plot(np.linspace(-0.2, 1.0, 121), lowProb_highLength[0], color='r', label='LowProb_highLength', linestyle='-')
plt.plot(np.linspace(-0.2, 1.0, 121), highProb_lowLength[0], color='b', label='HighProb_lowLength', linestyle='--')
plt.plot(np.linspace(-0.2, 1.0, 121), lowProb_lowLength[0], color='r', label='LowProb_lowLength', linestyle='--')
plt.legend()
plt.savefig(os.path.join(my_path,'ERFs','grandavg_interaction_clozeBYlength.png'))
plt.show()


# plot topograph of the effect: cloze x frequency
evoked_diff_highProb_freq = mne.combine_evoked([grandavg['highProb_lowFreq'], grandavg['highProb_highFreq']], weights=[1, -1])
evoked_diff_highFreq_prob = mne.combine_evoked([grandavg['lowProb_highFreq'], grandavg['highProb_highFreq']], weights=[1, -1])
evoked_diff_lowProb_freq = mne.combine_evoked([grandavg['lowProb_lowFreq'], grandavg['lowProb_highFreq']], weights=[1, -1])

evoked_diff_highProb_freq.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))
evoked_diff_highFreq_prob.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))
evoked_diff_lowProb_freq.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))

evoked_diff_highProb_freq.plot_topomap(times=[0.40],average=0.21, ch_type="mag", vlim=(-20, 20))
evoked_diff_highFreq_prob.plot_topomap(times=[0.40],average=0.21, ch_type="mag", vlim=(-20, 20))
evoked_diff_lowProb_freq.plot_topomap(times=[0.40],average=0.21, ch_type="mag", vlim=(-20, 20))

## cloze x length
evoked_diff_highProb_length = mne.combine_evoked([grandavg['highProb_lowLength'], grandavg['highProb_highLength']], weights=[1, -1])
evoked_diff_highLength_prob = mne.combine_evoked([grandavg['lowProb_highLength'], grandavg['highProb_highLength']], weights=[1, -1])
evoked_diff_lowProb_length = mne.combine_evoked([grandavg['lowProb_lowLength'], grandavg['lowProb_highLength']], weights=[1, -1])

evoked_diff_highProb_length.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))
evoked_diff_highLength_prob.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))
evoked_diff_lowProb_length.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-20, 20))

evoked_diff_highProb_length.plot_topomap(times=[0.30],average=0.21, ch_type="mag", vlim=(-20, 20))
evoked_diff_highLength_prob.plot_topomap(times=[0.30],average=0.21, ch_type="mag", vlim=(-20, 20))
evoked_diff_lowProb_length.plot_topomap(times=[0.30],average=0.21, ch_type="mag", vlim=(-20, 20))




## ------------------------------------------------
# rERF
variables = ['probs','length','frequency']
for var in variables:
    names = [var,'intercept']
    res = linear_regression(epochs, epochs.metadata[names], names=names)
    #for cond in names:
    #    res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))

reject_H0, fdr_pvals = fdr_correction(res[var].p_val.data) #sensor*time
evoked = res[var].beta #sensor*time
evoked.plot_image(mask=reject_H0, time_unit="s")





#############################################################################
# extract single-trial value
#############################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]

for file in file_lists[:2]:
    
    print(f'processing file: {file}')   
    
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    epochs.apply_baseline((-0.2, 0))
    epochs = epochs[5:] #remove the first 5 trials
    epochs = epochs.pick_channels(['MEG 065'])
    
    meta = epochs.metadata
    meta = meta.reset_index().rename(columns={'index':'epoch'})
    df = epochs.to_data_frame(time_format=None, scalings=dict(eeg=1e6, mag=1e15, grad=1e13), long_format=True)
        
    #get N400 amplitude
    filtered_df = df[(df['time'] >= 0.3) & (df['time'] <= 0.5)]
    df_N400 =  filtered_df.groupby(['condition', 'epoch', 'channel', 'ch_type'])['value'].mean().reset_index()
    
    df_full = df_N400.merge(meta[['epoch','words','probs','length','frequency',
                             'similarity_n_1','similarity_n_2','similarity_n_3',
                             'similarity_n_4','similarity_n_5']],
                       on='epoch')
    
    df_full.drop(columns=['condition','ch_type'], inplace=True)
    df_full['subID'] = file.split('_')[0]
    df_full['session'] = file.split('_')[1]
    df_full['task'] = file.split('_')[2][:-4]
    
    new_order = ['subID', 'session', 'task','epoch', 'channel', 'value', 'words', 'probs', 'length',
                 'frequency', 'similarity_n_1', 'similarity_n_2', 'similarity_n_3',
                 'similarity_n_4', 'similarity_n_5']

    df_full = df_full[new_order]
    
    fname = os.path.join(my_path,'rERF',file[:-4]+'.csv')    
    df_full.to_csv(fname, index=False)
    
