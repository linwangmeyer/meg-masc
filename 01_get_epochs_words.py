## Get epochs and words

import mne
import mne_bids
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import numpy as np
import pandas as pd
from tqdm import trange
import nltk
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib


#matplotlib.use("TkAgg")

my_path = '/cluster/home/lwang11/MASC-MEG'
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
    raw.load_data().filter(0.1, 30.0, n_jobs=1)
    
    # Identify back sensors by clicking on the channels
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
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)

    # get word information
    words = meta.query('kind=="word"').copy()
    meta_words = words[['story','story_uid','sound_id','sound','onset','duration','word_index','word']]
    
    # Word length
    meta_words['NumberOfLetters'] = meta_words['word'].apply(lambda x: len(x))

    # Word syntactic category
    meta_words['tokens'] = meta_words['word'].apply(lambda x: word_tokenize(x))
    meta_words['pos_tags'] = meta_words['tokens'].apply(lambda x: nltk.pos_tag(x))
    content_word_categories = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    meta_words['word_category'] = meta_words['pos_tags'].apply(lambda x: 'Content' if any(tag[1] in content_word_categories for tag in x) else 'Functional')

    # segment epochs for each word
    events = np.c_[
        meta_words.onset * raw.info["sfreq"], np.ones((len(meta_words), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=0.8,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta_words,
        preload=True,
        event_repeated="drop",
    )

    # baseline correction
    epochs.apply_baseline()

    return epochs


#####################
## Get epochs for all sessions and all tasks
subject='01'
for session in range(0):
    all_epochs = list()
    for task in range(4):
        raw = _get_raw(subject,session,task)
        raw_clean = _clean_raw(raw)
        epochs = _get_epochs(raw)
        all_epochs.append(epochs)
    epochs = mne.concatenate_epochs(all_epochs)
    epochs_fname = my_path + f"/Segments/sub{subject}_session{session}"
    epochs.save(epochs_fname,overwrite=True)

'''#####################
## Get epochs for each session and each task
subject='01'
for session in range(1):
    for task in range(4):
        raw = _get_raw(subject,session,task)
        raw_clean = _clean_raw(raw)
        epochs = _get_epochs(raw)
        all_epochs.append(epochs)
        epochs_fname = my_path + f"/Segments/sub{subject}_session{session}_task{task}"
        epochs.save(epochs_fname,overwrite=True)
'''

'''#################################################
## Get raw data for all sessions and all tasks
subject='01'
all_raw = list()
for session in range(1):
    for task in range(4):
        raw = _get_raw(subject,session,task)
        all_raw.append(raw)
    raws = mne.concatenate_raws(all_raw)
    raws_fname = my_path + f"/Raws/session{session}_sub{subject}.fif"
    raws.save(raws_fname,overwrite=True)

## Clean raw
subject='01'
for session in range(1):
    raws_fname = my_path + f"/Raws/session{session}_sub{subject}.fif"
    raw = mne.io.read_raw_fif(raws_fname)
    raw_clean = _clean_raw(raw)
    raw_clean_fname = my_path + f"/Raws_clean/session{session}_sub{subject}.fif"
    raw_clean.save(raw_clean,overwrite=True)

## For epochs
subject='01'
for session in range(1):
    raws_fname = my_path + f"/Raws_clean/session{session}_sub{subject}.fif"
    raw_clean = mne.io.read_raw_fif(raws_fname)
    epochs = _get_epochs(raw)
    epochs_fname = my_path + f"/Segments/session{session}_sub{subject}"
    epochs.save(epochs_fname,overwrite=True)
'''