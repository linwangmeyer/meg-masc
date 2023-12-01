# calculate evoked responses
import json
import mne
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from mne.viz import plot_compare_evokeds
from scipy.stats import pearsonr
import matplotlib
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
    reject = dict(mag=4e-12)
    epochs = mne.EpochsArray(epochs.get_data(), epochs.info,reject=reject, tmin=-0.2, reject_tmin=0, reject_tmax=1.0, baseline=(-0.2, 0), metadata=epochs.metadata)
    
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    
    median_prob = metadata['probs'].median()
    epochs_low = epochs[metadata['probs'] < median_prob]
    epochs_high = epochs[metadata['probs'] >= median_prob]

    evokeds_low = epochs_low.average().apply_baseline((-0.2, 0))
    evokeds_high = epochs_high.average().apply_baseline((-0.2, 0))

    evoked_fname = os.path.join(my_path,'ERFs','cloze_'+file)
    mne.write_evokeds(evoked_fname,[evokeds_low, evokeds_high],overwrite=True)

################################################################
# evoked activity to different conditions of features: word2vec
################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
for file in file_lists:
    
    print(f'processing file: {file}')   
        
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    reject = dict(mag=4e-12)
    epochs = mne.EpochsArray(epochs.get_data(), epochs.info,reject=reject, tmin=-0.2, reject_tmin=0, reject_tmax=1.0, baseline=(-0.2, 0), metadata=epochs.metadata)
    
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    
    median_sim_n_1 = metadata['similarity_n_1'].median()    
    epochs_low = epochs[metadata['similarity_n_1'] < median_sim_n_1]
    epochs_high = epochs[metadata['similarity_n_1'] >= median_sim_n_1]

    evokeds_low = epochs_low.average().apply_baseline((-0.2, 0))
    evokeds_high = epochs_high.average().apply_baseline((-0.2, 0))

    evoked_fname = os.path.join(my_path,'ERFs','word2vecN1_'+file)
    mne.write_evokeds(evoked_fname,[evokeds_low, evokeds_high],overwrite=True)
    
# plot ERFs of individual subject
#mne.viz.plot_compare_evokeds([evokeds_low, evokeds_high], picks=['MEG 054'])
#mne.viz.plot_compare_evokeds([evokeds_low, evokeds_high], picks=['MEG 065'])


# ------------ Grand average ------------------
file_lists = [file for file in os.listdir(my_path+'ERFs/') if file.endswith(".fif")]
all_low = []
all_high = []
for file in file_lists:
    if file.startswith('cloze_sub'):
        read_fname = os.path.join(my_path,'ERFs',file)
        print(f'processing file: {file}')
        
        evokeds_low, evokeds_high = mne.read_evokeds(read_fname)    
        all_low.append(evokeds_low)
        all_high.append(evokeds_high)
    
grandavg_low = mne.grand_average(all_low,interpolate_bads=False,drop_bads=False)
grandavg_high = mne.grand_average(all_high,interpolate_bads=False,drop_bads=False)
evoked_fname = os.path.join(my_path,'ERFs','cloze_grandavg.fif')
mne.write_evokeds(evoked_fname,[grandavg_low, grandavg_high],overwrite=True)

#word2vec effect
file_lists = [file for file in os.listdir(my_path+'ERFs/') if file.endswith(".fif")]
all_low = []
all_high = []
for file in file_lists:
    if file.startswith('word2vecN1_sub'):
        read_fname = os.path.join(my_path,'ERFs',file)
        print(f'processing file: {file}')
        
        evokeds_low, evokeds_high = mne.read_evokeds(read_fname)    
        all_low.append(evokeds_low)
        all_high.append(evokeds_high)
    
grandavg_low = mne.grand_average(all_low,interpolate_bads=False,drop_bads=False)
grandavg_high = mne.grand_average(all_high,interpolate_bads=False,drop_bads=False)
evoked_fname = os.path.join(my_path,'ERFs','word2vecN1_grandavg.fif')
mne.write_evokeds(evoked_fname,[grandavg_low, grandavg_high],overwrite=True)

##################################################################################
# ------------ Plot Grand average ------------------
my_path = r'S:/USERS/Lin/MASC-MEG/'
evoked_fname = os.path.join(my_path,'ERFs','cloze_grandavg.fif')
grandavg = mne.read_evokeds(evoked_fname)
grandavg_low = grandavg[0]
grandavg_high = grandavg[1]

## Plot ERFs: Cloze
plot_sensor = mne.pick_channels(grandavg_low.ch_names,['MEG 065'])
lowProb = grandavg_low.get_data()[plot_sensor,:]
highProb = grandavg_high.get_data()[plot_sensor,:]
plt.plot(np.linspace(-0.2, 1.0, 121), lowProb[0], color='r', label='LowProb', linestyle='-')
plt.plot(np.linspace(-0.2, 1.0, 121), highProb[0], color='b', label='HighProb', linestyle='-')
plt.legend()
plt.savefig(os.path.join(my_path,'ERFs','grandavg_cloze.png'))
plt.show()

# Plot Topos: cloze
evoked_diff = mne.combine_evoked([grandavg_low, grandavg_high], weights=[1, -1])
mask = np.zeros(evoked_diff.data.shape, dtype="bool")
highlightsensors = ['MEG 084','MEG 134','MEG 065','MEG 053','MEG 122','MEG 123','MEG 140','MEG 160']
picksensor = mne.pick_channels(evoked_diff.ch_names,highlightsensors)
mask[picksensor,:]=True
mask_params = dict(markersize=10, markerfacecolor="y")
evoked_diff.info['bads']=[]
evoked_diff.plot_topomap(times=[0.10, 0.40], average=0.21, ch_type="mag", vlim=(-15, 15), colorbar=False,mask=mask,mask_params=mask_params)

# Plot ERFs of 'bad' sensors
badsensor = mne.pick_channels(grandavg_low.ch_names,grandavg_low.info['bads'])
plotsensor = np.append(badsensor,52)
plt.plot(grandavg_low.get_data()[plotsensor,:].T)
plt.show()


# repair bad sensors (doesn't work)
evoked_int = grandavg_low.interpolate_bads(reset_bads=True,exclude=[], origin='auto')

# check sensor layout
layout = mne.channels.find_layout(grandavg_low.info, ch_type='meg')
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
    reject = dict(mag=4e-12)
    epochs = mne.EpochsArray(epochs.get_data(), epochs.info,reject=reject, tmin=-0.2, reject_tmin=0, reject_tmax=1.0, baseline=(-0.2, 0), metadata=epochs.metadata)    
        
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    
    # Calculate median values for each variable
    median_prob = metadata['probs'].median()
    median_freq = metadata['frequency'].median()
    median_length = metadata['length'].median()
    median_sim_n_1 = metadata['similarity_n_1'].median()
    median_sim_n_2 = metadata['similarity_n_2'].median()
    median_sim_n_3 = metadata['similarity_n_3'].median()

    # Define conditions
    conditions = [
        ((metadata['probs'] < median_prob) & (metadata['frequency'] < median_freq), 'lowProb_lowFreq'),
        ((metadata['probs'] >= median_prob) & (metadata['frequency'] < median_freq), 'highProb_lowFreq'),
        ((metadata['probs'] < median_prob) & (metadata['frequency'] >= median_freq), 'lowProb_highFreq'),
        ((metadata['probs'] >= median_prob) & (metadata['frequency'] >= median_freq), 'highProb_highFreq'),

        ((metadata['probs'] < median_prob) & (metadata['length'] < median_length), 'lowProb_lowLength'),
        ((metadata['probs'] >= median_prob) & (metadata['length'] < median_length), 'highProb_lowLength'),
        ((metadata['probs'] < median_prob) & (metadata['length'] >= median_length), 'lowProb_highLength'),
        ((metadata['probs'] >= median_prob) & (metadata['length'] >= median_length), 'highProb_highLength'),

        ((metadata['probs'] < median_prob) & (metadata['similarity_n_1'] < median_sim_n_1), 'lowProb_lowSimN1'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_1'] < median_sim_n_1), 'highProb_lowSimN1'),
        ((metadata['probs'] < median_prob) & (metadata['similarity_n_1'] >= median_sim_n_1), 'lowProb_highSimN1'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_1'] >= median_sim_n_1), 'highProb_highSimN1'),

        ((metadata['probs'] < median_prob) & (metadata['similarity_n_2'] < median_sim_n_2), 'lowProb_lowSimN2'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_2'] < median_sim_n_2), 'highProb_lowSimN2'),
        ((metadata['probs'] < median_prob) & (metadata['similarity_n_2'] >= median_sim_n_2), 'lowProb_highSimN2'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_2'] >= median_sim_n_2), 'highProb_highSimN2'),
        
        ((metadata['probs'] < median_prob) & (metadata['similarity_n_3'] < median_sim_n_3), 'lowProb_lowSimN3'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_3'] < median_sim_n_3), 'highProb_lowSimN3'),
        ((metadata['probs'] < median_prob) & (metadata['similarity_n_3'] >= median_sim_n_3), 'lowProb_highSimN3'),
        ((metadata['probs'] >= median_prob) & (metadata['similarity_n_3'] >= median_sim_n_3), 'highProb_highSimN3')
    ]
    
    evokeds = {}
    for condition, label in conditions:
        epochs_condition = epochs[condition]
        evokeds[label] = epochs_condition.average().apply_baseline((-0.2, 0))
    
    all_evokeds.append(evokeds)

# calculate grandaverage
evokeds_by_condition = {}
for evokeds_dict in all_evokeds:
    for condition_name, evoked_obj in evokeds_dict.items():
        if condition_name not in evokeds_by_condition:
            evokeds_by_condition[condition_name] = []
        evokeds_by_condition[condition_name].append(evoked_obj)

grandavg = {}
for cond, evks in evokeds_by_condition.items():
    grandavg[cond] = mne.grand_average(evks,interpolate_bads=False,drop_bads=False)

for condition, evoked in grandavg.items():
    evoked.comment = condition
    evoked_fname = os.path.join(my_path,'ERFs',condition+'.fif')
    mne.write_evokeds(evoked_fname, evoked, overwrite=True)


############################################################
# Plot evoked effects

# load data
conditions = ['lowProb_lowFreq', 'highProb_lowFreq', 'lowProb_highFreq', 'highProb_highFreq',
              'lowProb_lowLength', 'highProb_lowLength', 'lowProb_highLength', 'highProb_highLength',
              'lowProb_lowSimN1', 'highProb_lowSimN1', 'lowProb_highSimN1', 'highProb_highSimN1',
              'lowProb_lowSimN2', 'highProb_lowSimN2', 'lowProb_highSimN2', 'highProb_highSimN2',
              'lowProb_lowSimN3', 'highProb_lowSimN3', 'lowProb_highSimN3', 'highProb_highSimN3']
grandavg = {}
for cond in conditions:
    evoked_fname = os.path.join(my_path,'ERFs',cond+'.fif')    
    grandavg[cond] = mne.read_evokeds(evoked_fname)[0]


# ---------------------------------------------------------
# ERF plots
# ---------------------------------------------------------
def plot_conditions_by_label(grandavg, sensors, my_path, label_tuples):
    '''grandavg: dictionary containing all grandavg plots
    sensors: select one MEG sensor to plot the ERFs
    my_path: place to save the data
    label_tuples: combination of conditions, representing cloze by lexical variables'''
    condition_data = {}
    for prob_label, lex_label in label_tuples:
        key = f'{prob_label}_{lex_label}'
        plot_sensor = mne.pick_channels(grandavg[key].ch_names, sensors)
        condition_data[key] = grandavg[key].get_data()[plot_sensor, :]

    for key, data in condition_data.items():
        prob_label, lex_label = key.split('_')
        linestyle = '-' if 'high' in lex_label else '--'
        color = 'b' if 'high' in prob_label else 'r'
        plt.plot(np.linspace(-0.2, 1.0, 121), data[0], label=f'{prob_label}_{lex_label}', color=color, linestyle=linestyle)
    plt.legend()
    plt.savefig(os.path.join(my_path,'ERFs','plots', 'grandavg_interaction_'+prob_label[-4:]+'BY'+lex_label[-4:]+'MEG'+sensors[0][-3:]+'.png'))
    plt.close()

# Plot all ERFs
my_path = r'S:/USERS/Lin/MASC-MEG/'
sensors = ['MEG 134']
label_tuples = [('highProb', 'highFreq'), ('lowProb', 'highFreq'), ('highProb', 'lowFreq'), ('lowProb', 'lowFreq')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('highProb', 'highLength'), ('lowProb', 'highLength'), ('highProb', 'lowLength'), ('lowProb', 'lowLength')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('highProb', 'highSimN1'), ('lowProb', 'highSimN1'), ('highProb', 'lowSimN1'), ('lowProb', 'lowSimN1')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('highProb', 'highSimN2'), ('lowProb', 'highSimN2'), ('highProb', 'lowSimN2'), ('lowProb', 'lowSimN2')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('highProb', 'highSimN3'), ('lowProb', 'highSimN3'), ('highProb', 'lowSimN3'), ('lowProb', 'lowSimN3')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)



# ---------------------------------------------------------
# topo plots
# ---------------------------------------------------------
# plot topograph of the effects
evoked_diff_highProb_Freq = mne.combine_evoked([grandavg['highProb_lowFreq'], grandavg['highProb_highFreq']], weights=[1, -1])
evoked_diff_lowProb_Freq = mne.combine_evoked([grandavg['lowProb_lowFreq'], grandavg['lowProb_highFreq']], weights=[1, -1])
evoked_diff_highFreq_Prob = mne.combine_evoked([grandavg['lowProb_highFreq'], grandavg['highProb_highFreq']], weights=[1, -1])
evoked_diff_lowFreq_Prob = mne.combine_evoked([grandavg['lowProb_lowFreq'], grandavg['highProb_lowFreq']], weights=[1, -1])

evoked_diff_highProb_SimN1 = mne.combine_evoked([grandavg['highProb_lowSimN1'], grandavg['highProb_highSimN1']], weights=[1, -1])
evoked_diff_lowProb_SimN1 = mne.combine_evoked([grandavg['lowProb_lowSimN1'], grandavg['lowProb_highSimN1']], weights=[1, -1])
evoked_diff_highSimN1_Prob = mne.combine_evoked([grandavg['lowProb_highSimN1'], grandavg['highProb_highSimN1']], weights=[1, -1])
evoked_diff_lowSimN1_Prob = mne.combine_evoked([grandavg['lowProb_lowSimN1'], grandavg['lowProb_highSimN1']], weights=[1, -1])

evoked_diff_highProb_SimN2 = mne.combine_evoked([grandavg['highProb_lowSimN2'], grandavg['highProb_highSimN2']], weights=[1, -1])
evoked_diff_lowProb_SimN2 = mne.combine_evoked([grandavg['lowProb_lowSimN2'], grandavg['lowProb_highSimN2']], weights=[1, -1])
evoked_diff_highSimN2_Prob = mne.combine_evoked([grandavg['lowProb_highSimN2'], grandavg['highProb_highSimN2']], weights=[1, -1])
evoked_diff_lowSimN2_Prob = mne.combine_evoked([grandavg['lowProb_lowSimN2'], grandavg['lowProb_highSimN2']], weights=[1, -1])

evoked_diff_highProb_SimN3 = mne.combine_evoked([grandavg['highProb_lowSimN3'], grandavg['highProb_highSimN3']], weights=[1, -1])
evoked_diff_lowProb_SimN3 = mne.combine_evoked([grandavg['lowProb_lowSimN3'], grandavg['lowProb_highSimN3']], weights=[1, -1])
evoked_diff_highSimN3_Prob = mne.combine_evoked([grandavg['lowProb_highSimN3'], grandavg['highProb_highSimN3']], weights=[1, -1])
evoked_diff_lowSimN3_Prob = mne.combine_evoked([grandavg['lowProb_lowSimN3'], grandavg['lowProb_highSimN3']], weights=[1, -1])

variables = [
    evoked_diff_highProb_Freq,
    evoked_diff_lowProb_Freq,
    evoked_diff_highFreq_Prob,
    evoked_diff_lowFreq_Prob,
    
    evoked_diff_highProb_SimN1,
    evoked_diff_lowProb_SimN1,
    evoked_diff_highSimN1_Prob,
    evoked_diff_lowSimN1_Prob,    
    
    evoked_diff_highProb_SimN2,
    evoked_diff_lowProb_SimN2,
    evoked_diff_highSimN2_Prob,
    evoked_diff_lowSimN2_Prob,
        
    evoked_diff_highProb_SimN3,
    evoked_diff_lowProb_SimN3,
    evoked_diff_highSimN3_Prob,
    evoked_diff_lowSimN3_Prob
]
for variable in variables:
    variable.info['bads'] = []

mask = np.zeros(evoked_diff_highFreq_Prob.data.shape, dtype="bool")
#highlightsensors = ['MEG 084','MEG 134','MEG 065','MEG 053','MEG 122','MEG 123','MEG 140','MEG 160']
highlightsensors = ['MEG 084','MEG 134','MEG 065','MEG 098','MEG 123','MEG 128','MEG 080','MEG 137']
picksensor = mne.pick_channels(evoked_diff_highFreq_Prob.ch_names,highlightsensors)
mask[picksensor,:]=True
mask_params = dict(markersize=10, markerfacecolor="y")
#evoked_diff_highProb_Freq.plot_topomap(times=[0.0, 0.10, 0.40, 0.60], ch_type="mag", vlim=(-15, 15),mask=mask,mask_params=mask_params)

## Cloze x freq
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
evoked_diff_highProb_Freq.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 0],colorbar=False,mask=mask,mask_params=mask_params)
axes[0, 0].set_title('High Prob: Low vs High Freq')
evoked_diff_lowProb_Freq.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[0, 1].set_title('Low Prob: Low vs High Freq')
evoked_diff_highFreq_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 0],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 0].set_title('High Freq: Low vs High Prob')
evoked_diff_lowFreq_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 1].set_title('Low Freq: Low vs High Prob')
plt.tight_layout()
plt.savefig(os.path.join(my_path,'ERFs','plots', 'topo_probBYfreq.png'))
plt.show()

## cloze x similarity N-1
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
evoked_diff_highProb_SimN1.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 0],colorbar=False,mask=mask,mask_params=mask_params,show_names=True)
axes[0, 0].set_title('High Prob: Low vs High SimN1')
evoked_diff_lowProb_SimN1.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[0, 1].set_title('Low Prob: Low vs High SimN1')
evoked_diff_highSimN1_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 0],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 0].set_title('High SimN1: Low vs High Prob')
evoked_diff_lowSimN1_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 1].set_title('Low SimN1: Low vs High Prob')
plt.tight_layout()
plt.savefig(os.path.join(my_path,'ERFs','plots', 'topo_probBYSimN1.png'))
plt.show()

## cloze x similarity N-2
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
evoked_diff_highProb_SimN2.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 0],colorbar=False,mask=mask,mask_params=mask_params,show_names=True)
axes[0, 0].set_title('High Prob: Low vs High SimN2')
evoked_diff_lowProb_SimN2.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[0, 1].set_title('Low Prob: Low vs High SimN2')
evoked_diff_highSimN2_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 0],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 0].set_title('High SimN2: Low vs High Prob')
evoked_diff_lowSimN2_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 1].set_title('Low SimN2: Low vs High Prob')
plt.tight_layout()
plt.savefig(os.path.join(my_path,'ERFs','plots', 'topo_probBYSimN2.png'))
plt.show()


## cloze x similarity N-3
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
evoked_diff_highProb_SimN3.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 0],colorbar=False,mask=mask,mask_params=mask_params,show_names=True)
axes[0, 0].set_title('High Prob: Low vs High SimN3')
evoked_diff_lowProb_SimN3.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[0, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[0, 1].set_title('Low Prob: Low vs High SimN3')
evoked_diff_highSimN3_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 0],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 0].set_title('High SimN3: Low vs High Prob')
evoked_diff_lowSimN3_Prob.plot_topomap(times=[0.40], average=0.21, ch_type="mag", vlim=(-15, 15), axes=axes[1, 1],colorbar=False,mask=mask,mask_params=mask_params)
axes[1, 1].set_title('Low SimN3: Low vs High Prob')
plt.tight_layout()
plt.savefig(os.path.join(my_path,'ERFs','plots', 'topo_probBYSimN3.png'))
plt.show()

'''
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
'''




#############################################################################
# extract single-trial value
#############################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]

for file in file_lists:
    
    print(f'processing file: {file}')   
    
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    reject = dict(mag=4e-12)
    epochs = mne.EpochsArray(epochs.get_data(), epochs.info,reject=reject, tmin=-0.2, reject_tmin=0, reject_tmax=1.0, baseline=(-0.2, 0), metadata=epochs.metadata)
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    epochs = epochs.pick_channels(['MEG 134'])
    
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
    



#############################################################################
# check relationship between cloze, lexico-semantic features and N400
#############################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'rERF/') if file.endswith(".csv")]
dfs = []
for file in file_lists:
    
    print(f'processing file: {file}')
    
    df = pd.read_csv(os.path.join(my_path,'rERF',file))
    dfs.append(df)

alldata = pd.concat(dfs[:],axis=0)

#--------------------------------------
# plot single-trial N400 modulated by similarity and cloze
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.scatterplot(x=alldata['similarity_n_1'], y=alldata['value'],alpha=0.5,s=5)
plt.title('similarity')
plt.subplot(1,2,2)
sns.scatterplot(x=alldata['probs'], y=alldata['value'],alpha=0.5,s=5)
plt.title('cloze')
plt.tight_layout()
plt.show()

#--------------------------------------
# average across epochs and plot relationships
df_cloze_simN1 = alldata.groupby(['probs','similarity_n_1'])['value'].mean().reset_index()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=df_cloze_simN1['similarity_n_1'], y=df_cloze_simN1['value'],alpha=0.5,s=5)
sns.regplot(x=df_cloze_simN1['similarity_n_1'], y=df_cloze_simN1['value'], scatter=False)
plt.subplot(1,2,2)
sns.scatterplot(x=df_cloze_simN1['probs'], y=df_cloze_simN1['value'],alpha=0.5,s=5)
sns.regplot(x=df_cloze_simN1['probs'], y=df_cloze_simN1['value'], scatter=False)
plt.show()

#--------------------------------------
# calculate correlations
corr, p_value = pearsonr(df_cloze_simN1['probs'], df_cloze_simN1['value'])
print(f"cloze effect: r: {corr:.2f}; p: {p_value:.4f}")

corr, p_value = pearsonr(df_cloze_simN1['similarity_n_1'], df_cloze_simN1['value'])
print(f"simN-1 effect r: {corr: .2f}; p: {p_value: .4f}")

#--------------------------------------
# fit a linear regression model
import statsmodels.api as sm
X = df_cloze_simN1[['similarity_n_1', 'probs']]
y = df_cloze_simN1['value']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#--------------------------------------
# check relationship between cloze and simN-1
df3 = alldata.groupby(['subID','session','task','probs','similarity_n_1','similarity_n_2'])['value'].mean().reset_index()
df3.columns.to_list()

corr, p_value = pearsonr(df3['probs'], df3['similarity_n_1'])
print(f"correlation between prob and word2vec n-1: r: {corr:.2f}; p: {p_value:.4f}")
corr, p_value = pearsonr(df3['probs'], df3['similarity_n_2'])
print(f"correlation between prob and word2vec n-2: r: {corr:.2f}; p: {p_value:.4f}")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
sns.regplot(x='probs', y='similarity_n_1',data=df3, ax=axes[0], scatter_kws={'s': 5})
axes[0].set_xlabel('probs')
axes[0].set_ylabel('similarity_n_1 mean value')
sns.regplot(x='probs', y='similarity_n_2',data=df3, ax=axes[1], scatter_kws={'s': 5})
axes[1].set_xlabel('probs')
axes[1].set_ylabel('similarity_n_2 mean value')
plt.tight_layout()
plt.show()

#--------------------------------------
# check charateristics of cloze and simN-1: two cloze bins
df3 = alldata.groupby(['subID','session','task','probs','similarity_n_1'])['value'].mean().reset_index()
df3.columns.to_list()

# cloze range: across subjects, sessions and tasks
df3['probs_median'] = df3.groupby(['subID', 'session', 'task'])['probs'].transform('median')
df3['probs_bins'] = 'low'
df3.loc[df3['probs'] >= df3['probs_median'], 'probs_bins'] = 'high'
results=df3.groupby(['subID', 'session', 'task', 'probs_bins'])['probs'].median().reset_index()
plt.figure(figsize=(10,5))
sns.boxplot(x=results['probs_bins'], y=results['probs'])
plt.xlabel = 'prob bins'
plt.ylabel = 'prob mean value'
plt.show()

# similarity range: across subjects, sessions and tasks
df3 = alldata.groupby(['subID','session','task','probs','similarity_n_1'])['value'].mean().reset_index()
df3['simN1_median'] = df3.groupby(['subID', 'session', 'task'])['similarity_n_1'].transform('median')
df3['simN1_bins'] = 'low'
df3.loc[df3['similarity_n_1'] >= df3['simN1_median'], 'simN1_bins'] = 'high'
results=df3.groupby(['subID', 'session', 'task', 'simN1_bins'])['similarity_n_1'].median().reset_index()
plt.figure(figsize=(10,5))
sns.boxplot(x=results['simN1_bins'], y=results['similarity_n_1'])
plt.xlabel = 'similarity_n_1 bins'
plt.ylabel = 'similarity_n_1 mean value'
plt.show()

#--------------------------------------
# check charateristics of cloze and simN-1: 10 bins
df3 = alldata.groupby(['subID','session','task','probs','similarity_n_1'])['value'].mean().reset_index()
df3['prob_bins'] = pd.qcut(df3['probs'], q=10)
df3['similarity_n_1_median'] = df3.groupby('prob_bins')['similarity_n_1'].transform('median')
df3['simN1_bins'] = 'low'
df3.loc[df3['similarity_n_1'] >= df3['similarity_n_1_median'], 'simN1_bins'] = 'high'

N400_bins = df3.groupby(['prob_bins','simN1_bins'])['value'].mean().reset_index(name='value')
pivot_df = N400_bins.pivot(index='prob_bins', columns='simN1_bins', values='value')
pivot_df['effect'] = pivot_df['low'] - pivot_df['high']
results = pivot_df.reset_index()
results['prob_range'] = results['prob_bins'].astype(str)
plt.figure(figsize=(10,10))
plt.bar(results['prob_range'], results['effect'])
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(my_path,'ERFs','plots', 'semN2effect_cloze_bars.png'))
plt.show()

sim_bins = df3.groupby(['prob_bins','simN1_bins'])['similarity_n_1'].mean().reset_index(name='similarity_n_1')
pivot_df = sim_bins.pivot(index='prob_bins', columns='simN1_bins', values='similarity_n_1')
pivot_df['effect'] = pivot_df['low'] - pivot_df['high']
results = pivot_df.reset_index()
results['prob_range'] = results['prob_bins'].astype(str)
plt.figure(figsize=(10,10))
plt.bar(results['prob_range'], results['effect'])
plt.xticks(rotation=45, ha='right')
plt.savefig(os.path.join(my_path,'ERFs','plots', 'semN1values_cloze_bars.png'))
plt.show()


#--------------------------------------
# check relationship between N400 and simN-1 for each of the 10 cloze bins
df3 = alldata.groupby(['subID','session','task','probs','similarity_n_1'])['value'].mean().reset_index()
df3['prob_bins'] = pd.qcut(df3['probs'], q=10)
df3_sorted = df3.sort_values('prob_bins')
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
scatter_plot = sns.lmplot(x='similarity_n_1', y='probs', hue='prob_bins', data=df3_sorted, ci=None, height=6, aspect=1.5, scatter_kws={'s': 5})
plt.xlabel('Similarity_n_1')
plt.ylabel('Probs')
plt.title('Scatter Plot with Regression Lines for Each prob_bins')
plt.legend(title='Prob Bins', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()




#############################################################################
# evoked activity to wor2vec x cloze interaction: 4 cloze levels
#############################################################################
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
all_evokeds = []

for file in file_lists:
    
    print(f'processing file: {file}')   
        
    # get clean epochs of experimental conditions
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)
    reject = dict(mag=4e-12)
    epochs = mne.EpochsArray(epochs.get_data(), epochs.info,reject=reject, tmin=-0.2, reject_tmin=0, reject_tmax=1.0, baseline=(-0.2, 0), metadata=epochs.metadata)    
        
    epochs = epochs[5:] #remove the first 5 trials
    metadata = epochs.metadata
    
    # Calculate median values for each variable
    prob_0_10 = (metadata['probs'] > 0) & (metadata['probs'] < 0.10)
    prob_10_25 = (metadata['probs'] >= 0.10) & (metadata['probs'] < 0.25)
    prob_25_45 = (metadata['probs'] >= 0.25) & (metadata['probs'] < 0.45)
    prob_45_100 = (metadata['probs'] >= 0.45) & (metadata['probs'] < 1.0)
    
    median_sim_n_1 = metadata['similarity_n_1'].median()
    median_sim_n_2 = metadata['similarity_n_2'].median()
    median_sim_n_3 = metadata['similarity_n_3'].median()

    # Define conditions
    conditions = [
        (prob_0_10 & (metadata['similarity_n_1'] < median_sim_n_1), 'prob_0_10_lowSimN1'),
        (prob_10_25 & (metadata['similarity_n_1'] < median_sim_n_1), 'prob_10_25_lowSimN1'),
        (prob_25_45 & (metadata['similarity_n_1'] < median_sim_n_1), 'prob_25_45_lowSimN1'),
        (prob_45_100 & (metadata['similarity_n_1'] < median_sim_n_1), 'prob_45_100_lowSimN1'),
        
        (prob_0_10 & (metadata['similarity_n_1'] >= median_sim_n_1), 'prob_0_10_highSimN1'),
        (prob_10_25 & (metadata['similarity_n_1'] >= median_sim_n_1), 'prob_10_25_highSimN1'),
        (prob_25_45 & (metadata['similarity_n_1'] >= median_sim_n_1), 'prob_25_45_highSimN1'),
        (prob_45_100 & (metadata['similarity_n_1'] >= median_sim_n_1), 'prob_45_100_highSimN1'),
        
        (prob_0_10 & (metadata['similarity_n_2'] < median_sim_n_2), 'prob_0_10_lowSimN2'),
        (prob_10_25 & (metadata['similarity_n_2'] < median_sim_n_2), 'prob_10_25_lowSimN2'),
        (prob_25_45 & (metadata['similarity_n_2'] < median_sim_n_2), 'prob_25_45_lowSimN2'),
        (prob_45_100 & (metadata['similarity_n_2'] < median_sim_n_2), 'prob_45_100_lowSimN2'),
        
        (prob_0_10 & (metadata['similarity_n_2'] >= median_sim_n_2), 'prob_0_10_highSimN2'),
        (prob_10_25 & (metadata['similarity_n_2'] >= median_sim_n_2), 'prob_10_25_highSimN2'),
        (prob_25_45 & (metadata['similarity_n_2'] >= median_sim_n_2), 'prob_25_45_highSimN2'),
        (prob_45_100 & (metadata['similarity_n_2'] >= median_sim_n_2), 'prob_45_100_highSimN2'),
        
        (prob_0_10 & (metadata['similarity_n_3'] < median_sim_n_3), 'prob_0_10_lowSimN3'),
        (prob_10_25 & (metadata['similarity_n_3'] < median_sim_n_3), 'prob_10_25_lowSimN3'),
        (prob_25_45 & (metadata['similarity_n_3'] < median_sim_n_3), 'prob_25_45_lowSimN3'),
        (prob_45_100 & (metadata['similarity_n_3'] < median_sim_n_3), 'prob_45_100_lowSimN3'),
        
        (prob_0_10 & (metadata['similarity_n_3'] >= median_sim_n_3), 'prob_0_10_highSimN3'),
        (prob_10_25 & (metadata['similarity_n_3'] >= median_sim_n_3), 'prob_10_25_highSimN3'),
        (prob_25_45 & (metadata['similarity_n_3'] >= median_sim_n_3), 'prob_25_45_highSimN3'),
        (prob_45_100 & (metadata['similarity_n_3'] >= median_sim_n_3), 'prob_45_100_highSimN3')
    ]
    
    evokeds = {}
    for condition, label in conditions:
        epochs_condition = epochs[condition]
        evokeds[label] = epochs_condition.average().apply_baseline((-0.2, 0))
    
    all_evokeds.append(evokeds)

# calculate grandaverage
evokeds_by_condition = {}
for evokeds_dict in all_evokeds:
    for condition_name, evoked_obj in evokeds_dict.items():
        if condition_name not in evokeds_by_condition:
            evokeds_by_condition[condition_name] = []
        evokeds_by_condition[condition_name].append(evoked_obj)

grandavg = {}
for cond, evks in evokeds_by_condition.items():
    grandavg[cond] = mne.grand_average(evks,interpolate_bads=False,drop_bads=False)

for condition, evoked in grandavg.items():
    evoked.comment = condition
    evoked_fname = os.path.join(my_path,'ERFs',condition+'.fif')
    mne.write_evokeds(evoked_fname, evoked, overwrite=True)


############################################################
# Plot evoked effects

# load data
conditions = ['prob_0_10_lowSimN1', 'prob_10_25_lowSimN1', 'prob_25_45_lowSimN1', 'prob_45_100_lowSimN1',
 'prob_0_10_highSimN1', 'prob_10_25_highSimN1', 'prob_25_45_highSimN1', 'prob_45_100_highSimN1',
 'prob_0_10_lowSimN2', 'prob_10_25_lowSimN2', 'prob_25_45_lowSimN2', 'prob_45_100_lowSimN2', 
 'prob_0_10_highSimN2', 'prob_10_25_highSimN2', 'prob_25_45_highSimN2', 'prob_45_100_highSimN2', 
 'prob_0_10_lowSimN3', 'prob_10_25_lowSimN3', 'prob_25_45_lowSimN3', 'prob_45_100_lowSimN3', 
 'prob_0_10_highSimN3', 'prob_10_25_highSimN3', 'prob_25_45_highSimN3', 'prob_45_100_highSimN3']
grandavg = {}
for cond in conditions:
    evoked_fname = os.path.join(my_path,'ERFs',cond+'.fif')    
    grandavg[cond] = mne.read_evokeds(evoked_fname)[0]


# ---------------------------------------------------------
# ERF plots
# ---------------------------------------------------------
def plot_conditions_by_label(grandavg, sensors, my_path, label_tuples):
    '''grandavg: dictionary containing all grandavg plots
    sensors: select one MEG sensor to plot the ERFs
    my_path: place to save the data
    label_tuples: combination of conditions, representing cloze by lexical variables'''
    condition_data = {}
    for prob_label, lex_label in label_tuples:
        key = f'{prob_label}_{lex_label}'
        plot_sensor = mne.pick_channels(grandavg[key].ch_names, sensors)
        condition_data[key] = grandavg[key].get_data()[plot_sensor, :]

    prob_label_to_linestyle = {
        'prob_0_10': '-',
        'prob_10_25': '--',
        'prob_25_45': '-.',
        'prob_45_100': ':'}
    
    for key, data in condition_data.items():
        last_split = key.rfind('_')
        prob_label = key[:last_split]
        lex_label = key[last_split+1:]
        linestyle = prob_label_to_linestyle.get(prob_label, '-')  # Default to '-' if not found
        color = 'b' if 'high' in lex_label else 'r'
        plt.plot(np.linspace(-0.2, 1.0, 121), data[0], label=f'{prob_label}_{lex_label}', color=color, linestyle=linestyle)
    plt.legend()
    plt.savefig(os.path.join(my_path,'ERFs','plots', 'grandavg_interaction_prob'+prob_label[-4:]+'BY'+lex_label[-4:]+'_MEG'+sensors[0][-3:]+'.png'))
    plt.close()

# Plot all ERFs
my_path = r'S:/USERS/Lin/MASC-MEG/'
sensors = ['MEG 134']
label_tuples = [('prob_10_25', 'highSimN1'), ('prob_0_10', 'highSimN1'), ('prob_10_25', 'lowSimN1'), ('prob_0_10', 'lowSimN1')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('prob_45_100', 'highSimN1'), ('prob_25_45', 'highSimN1'), ('prob_45_100', 'lowSimN1'), ('prob_25_45', 'lowSimN1')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)

label_tuples = [('prob_10_25', 'highSimN2'), ('prob_0_10', 'highSimN2'), ('prob_10_25', 'lowSimN2'), ('prob_0_10', 'lowSimN2')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)
label_tuples = [('prob_45_100', 'highSimN2'), ('prob_25_45', 'highSimN2'), ('prob_45_100', 'lowSimN2'), ('prob_25_45', 'lowSimN2')]
plot_conditions_by_label(grandavg, sensors, my_path, label_tuples)


