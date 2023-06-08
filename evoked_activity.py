## Test evoked activity

import mne
import pandas as pd
import numpy as np

my_path = '/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
epochs_path = my_path + "/bids_anonym"
subject='01'
epochs = mne.read_epochs(epochs_path)

###########################################################
# Check evoked activity modulated by word length
epochs.metadata['NumberOfLetters'] = epochs.metadata['word'].apply(lambda x: len(x))
evokeds = dict()
query = "NumberOfLetters == {}"
for n_letters in epochs.metadata["NumberOfLetters"].unique():
    evokeds[str(n_letters)] = epochs[query.format(n_letters)].average()
mne.viz.plot_compare_evokeds(evokeds, cmap=("word length", "viridis"), picks="MEG 011")

