## Test evoked activity
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#matplotlib.use("TkAgg")

my_path = my_path = r'\\rstore.uit.tufts.edu\as_rsch_NCL02$\USERS\Lin\MASC-MEG'
subject='01'
epochs_fname = my_path + f"/Segments/sub{subject}"
epochs = mne.read_epochs(epochs_fname)

###########################################################
# Check evoked activity modulated by word length for content words
new_epochs = epochs[epochs.metadata['word_category']=='Content']
bin_edges = [1,4,8,17]
new_epochs.metadata['NumberOfLetters_bin'] = pd.cut(new_epochs.metadata['NumberOfLetters'], bins=bin_edges, labels=False, include_lowest=True) + 1
new_epochs.metadata['NumberOfLetters_bin'].value_counts()
evokeds = dict()
query = "NumberOfLetters_bin == {}"
picks=['MEG 101','MEG 120']
for n_letters in new_epochs.metadata["NumberOfLetters_bin"].unique():
    evokeds[str(n_letters)] = new_epochs[query.format(n_letters)].average().apply_baseline((None, 0))
mne.viz.plot_compare_evokeds(evokeds, cmap=("word length", "viridis"), picks=picks)

# plot each condition
evokeds["2"].plot(picks="mag", spatial_colors=True, gfp=True)
evokeds["2"].plot_joint()

# check sensor layout
layout = mne.channels.find_layout(new_epochs.info, ch_type='meg')
layout.plot()


###########################################################
# Check evoked activity modulated by word frequency for content words
new_epochs = epochs[epochs.metadata['word_category']=='Content']
new_epochs.metadata['word_freq'].hist()

bin_edges = [1,4,8,17]
new_epochs.metadata['word_freq'] = pd.cut(new_epochs.metadata['word_freq'], bins=bin_edges, labels=False, include_lowest=True) + 1
new_epochs.metadata['word_freq'].value_counts()
evokeds = dict()
query = "word_freq == {}"
picks=['MEG 101','MEG 120']
for n_letters in new_epochs.metadata["word_freq"].unique():
    evokeds[str(n_letters)] = new_epochs[query.format(n_letters)].average().apply_baseline((None, 0))
mne.viz.plot_compare_evokeds(evokeds, cmap=("word length", "viridis"), picks=picks)

# plot each condition
evokeds["2"].plot(picks="mag", spatial_colors=True, gfp=True)
evokeds["2"].plot_joint()

# check sensor layout
layout = mne.channels.find_layout(new_epochs.info, ch_type='meg')
layout.plot()
