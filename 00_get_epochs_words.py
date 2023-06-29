## Get epochs and words

import mne
import mne_bids
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
import os

matplotlib.use("TkAgg")
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
sub_path = my_path + "/bids_anonym"
subjects = pd.read_csv(sub_path+"/participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values

def _get_raw(subject,session,task):
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=str(session),
        task=str(task),
        datatype="meg",
        root=sub_path,
    )
    raw = mne_bids.read_raw_bids(bids_path)
    raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)

    return raw


def _clean_raw(raw):
    raw.load_data().filter(0.1, 30.0, n_jobs=2)
    
    # Identify back sensors by clicking on the channels
    bad_chan = ['MEG 067', 'MEG 079', 'MEG 148', 'MEG 183'] #identified based on visual inspections
    raw.info['bads'] = bad_chan
    raw.plot()
    print(raw.info['bads'])
    plt.show()
    
    # ICA
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    ica.plot_sources(raw, show_scrollbars=False)
    plt.show()
    ica.plot_components()
    plt.show()
    
    # Identify and remove ICA components
    cmp_pick = input("Enter the component numbers to remove (e.g., 0, 1): ")
    cmp_pick = [int(cmp.strip()) for cmp in cmp_pick.split(",")] if cmp_pick else []

    ica.plot_properties(raw, picks=cmp_pick)
    plt.show()
    
    ica.exclude=cmp_pick
    ica.apply(raw)
    
    return raw


def _get_epochs(raw):    
    # Preproc annotations to prepare for event info
    meta = list()
    for annot in raw.annotations:
        if annot['description'] != 'BAD boundary' and annot['description'] != 'EDGE boundary':
            d = eval(annot.pop("description"))
            for k, v in annot.items():
                assert k not in d.keys()
                d[k] = v
            meta.append(d)
    meta = pd.DataFrame(meta)

    # get word information
    words = meta.query('kind=="word"').copy()
    meta_words = words[['story','story_uid','sound_id','sound','onset','duration','word_index','word']]
    word_list = meta_words['word'].tolist()
    df_words = pd.DataFrame({'word': word_list})
    
    # segment epochs for each word
    events = np.c_[
        meta_words.onset * raw.info["sfreq"], np.ones((len(meta_words), 2))
    ].astype(int)

    reject_criteria = dict(mag=4000e-15)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=0.8,
        reject=reject_criteria,
        decim=10,
        baseline=(-0.2, 0.0),
        preload=True,
        event_repeated="drop",
    )
    keeptrl = [itrl for itrl, trl in enumerate(epochs.drop_log) if len(trl)==0]
    keepmeta = meta_words.iloc[keeptrl]
    epochs.metadata = keepmeta
    
    df_words = df_words.iloc[keeptrl]
    return epochs,df_words

#################################################
## Get epochs for each session and each task
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'

subjects = range(1,2) #sub03, 12, 16, 20, 21, don't have session1; 25: no session0
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(1):
        for task in range(1):
            raw = _get_raw(subject,session,task)
            
            raw_clean = _clean_raw(raw)
            
            epochs,df_words = _get_epochs(raw_clean)
            epochs_fname = my_path + f"/segments/sub{subject}_session{session}_task{task}.fif"
            epochs.save(epochs_fname,overwrite=True)
            
            word_fname = my_path + f"/segments/words_sub{subject}_session{session}_task{task}.csv"
            df_words.to_csv(word_fname,index=False)
            
            del raw, raw_clean, epochs, df_words
            
'''
#################################################
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'

subjects = range(6,28)
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(1):
        raws_fname = my_path + f"/Raws/session{session}_sub{subject}.fif"
        if os.path.exists(raws_fname):
            raw = mne.io.read_raw_fif(raws_fname)
            raw_clean = _clean_raw(raw)
            raw_clean_fname = my_path + f"/Raws_clean/session{session}_sub{subject}.fif"
            raw_clean.save(raw_clean_fname,overwrite=True)
            print(f'------session{session} of subject{subject} is done!')
            del raw, raw_clean
            print(f"File {raws_fname} does not exist.")
        else:
            print(f"File {raws_fname} does not exist.")
## For epochs
subjects = range(1,28)
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(2):
        raw_clean_fname = my_path + f"/Raws_clean/session{session}_sub{subject}.fif"
        if os.path.exists(raw_clean_fname):
            raw_clean = mne.io.read_raw_fif(raw_clean_fname)
            epochs = _get_epochs(raw_clean)
            epochs_fname = my_path + f"/Segments/session{session}_sub{subject}"
            epochs.save(epochs_fname,overwrite=True)
            print(f'------session{session} of subject{subject} is done!')
        else:
            print(f"File {raw_clean_fname} does not exist.")
'''