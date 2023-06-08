## Get features related to words

import mne
import pandas as pd
import numpy as np

my_path = '/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
epochs_path = my_path + "/bids_anonym"
subject='01'
epochs = mne.read_epochs(epochs_path)

#################################################
data = epochs._get_data() #trial*chan*time
words = epochs.metadata['word']

