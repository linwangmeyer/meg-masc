## Get epochs and words
import re
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
from difflib import ndiff

matplotlib.use("TkAgg")
my_path = r'S:/USERS/Lin/MASC-MEG/'
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
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
    words = meta.loc[(meta['condition']=='sentence') & (meta['kind']=='word')]
    #words = meta.loc[meta['kind']=='word'] #the old function that includes random words
    meta_words = words[['story','story_uid','sound_id','sound','onset','duration','word_index','word']]
    word_list = meta_words['word'].tolist()
    df_words = pd.DataFrame({'word': word_list})
    
    # segment epochs for each word
    events = np.c_[
        meta_words.onset * raw.info["sfreq"], np.ones((len(meta_words), 2))
    ].astype(int)

    #reject_criteria = dict(mag=4000e-15)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.2,
        tmax=1.0,
        #reject=reject_criteria,
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
my_path = r'S:/USERS/Lin/MASC-MEG/'
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
     
subjects = range(1,27) #sub03, 12, 16, 20, 21, don't have session1; 25: no session0
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(1):
        for task in range(1):
            try:
                raw = _get_raw(subject,session,task)                
                raw_clean = _clean_raw(raw)
                raw_clean_fname = my_path + f"/Raws_clean/sub{subject}_session{session}_task{task}.fif"
                raw_clean.save(raw_clean_fname,overwrite=True)
                
                epochs,df_words = _get_epochs(raw_clean)                
                epochs_fname = my_path + f"/segments_cw/session{session}_sub{subject}"
                epochs.save(epochs_fname,overwrite=True)
                word_fname = my_path + f"/segments/words_session{session}_sub{subject}.csv"
                df_words.to_csv(word_fname,index=False)
                print(f'------session{session} of subject{subject} is done!')
                
            except FileNotFoundError as e:
                print(f"FileNotFoundError for subject {subject}, session {session}, task {task}: {e}")
                continue

##--------------------------------------------------------------------------------------
# re-run epochs: following the '_clean_raw' function
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_names = os.listdir(my_path + 'Raws_clean/')
for file_name in file_names:
    raw_clean_fname = my_path + f'Raws_clean/{file_name}'
    raw_clean = mne.io.read_raw_fif(raw_clean_fname)
    epochs,df_words = _get_epochs(raw_clean)
    epochs_fname = my_path + f"/segments/{file_name}"
    epochs.save(epochs_fname,overwrite=True)
    
    word_fname = my_path + f"/segments/words_{file_name[:-3]}.csv"
    df_words.to_csv(word_fname,index=False)
    

# Note: the following scripts are not necessary to run after updating function '_get_epochs'
#################################################
##--------------------------------------------------------------------------------------
# to make up previous mistake in defining epochs (including random words)
##--------------------------------------------------------------------------------------
def _get_experimental_items(raw):
    # Only include items that are in the sentence conditions (i.e. remove random word lists)
    meta = list()
    for annot in raw.annotations:
        if annot['description'] != 'BAD boundary' and annot['description'] != 'EDGE boundary':
            d = eval(annot.pop("description"))
            for k, v in annot.items():
                assert k not in d.keys()
                d[k] = v
            meta.append(d)
    meta = pd.DataFrame(meta)

    # get word information: remove words in random word list
    words = meta.loc[(meta['condition']=='sentence') & (meta['kind']=='word')]
    meta_words = words[['story','story_uid','sound_id','sound','onset','duration','word_index','word']]
    word_list = meta_words['word'].tolist()
    df_words = pd.DataFrame({'word': word_list})
    return df_words




def find_epoch_index(df_words,df_words_new):
    '''Identify items that were removed due to either not-sentence-condition or bad epochs
    df_words: dataframe that contains all words identified in the annotations
    df_words_new: dataframe that contains only words that have good signals and in the sentence condition'''
    
    all_words = df_words['word'].tolist()
    new_words = df_words_new['word'].tolist()
    extra_word_indices = []
    k=0
    for i, word in enumerate(all_words):
        if word in new_words[k:i+1]:
            k=k+1
        elif word not in new_words[k:i + 1]:
            extra_word_indices.append(i)
    all_indices = range(len(all_words))
    kept_indices = [index for index in all_indices if index not in extra_word_indices]
    return kept_indices


##--------------------------------------------------------------------------------------
# Update epochs and save them in a new folder
my_path = r'S:/USERS/Lin/MASC-MEG/'
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'

# get the word list in the experimental conditions (no random words)
subjects = range(1,27) #sub03, 12, 16, 20, 21, don't have session1; 25: no session0
for i in subjects:
    subject = str(i).zfill(2)
    for session in range(2):
        for task in range(4):
            try:
                raw = _get_raw(subject,session,task)
                df_words = _get_experimental_items(raw)
                
                word_fname = my_path + f"/Raws_clean/words_sub{subject}_session{session}_task{task}.csv"
                df_words.to_csv(word_fname,index=False)
            except FileNotFoundError as e:
                print(f"FileNotFoundError for subject {subject}, session {session}, task {task}: {e}")
                continue
       
##------------------------------------------------------------------------------------------------
# get epochs that only contain words in the story (i.e. remove random words)
# by comparing the words associated with the epochs
my_path = r'S:/USERS/Lin/MASC-MEG/'
file_lists = [file for file in os.listdir(my_path+'segments/') if file.endswith(".fif")]
for file in file_lists:
    
    print(f'processing file: {file}')
    
    epochs_fname = my_path + f"/segments/{file}"
    epochs = mne.read_epochs(epochs_fname)

    #get the critical words: with random word list
    word_fname = my_path + f"/segments/words_{file[:-4]}.csv"
    df_words = pd.read_csv(word_fname)

    #get the critical words: without random word list
    word_fname = my_path + f"/Raws_clean/words_{file[:-4]}.csv"
    df_words_new = pd.read_csv(word_fname)

    # get the clean epochs
    kept_indices = find_epoch_index(df_words,df_words_new)
    epochs_new= epochs[kept_indices]
    epochs_new.metadata.reset_index(drop=True,inplace=True)
    epochs_fname = my_path + f"/segments_cw/{file}"
    epochs_new.save(epochs_fname,overwrite=True)




'''
def find_epoch_index(df_words,df_words_new):
    '''Identify items that were removed due to either not-sentence-condition or bad epochs
    df_words: dataframe that contains all words identified in the annotations
    df_words_new: dataframe that contains only words that have good signals and in the sentence condition'''
    removed_indices = set()
    while df_words_new.shape[0] < df_words.shape[0]:
        for ind in range(df_words_new.shape[0]):
            if not df_words_new.iloc[ind].equals(df_words.iloc[ind]):
                removed_indices.add(df_words.index[ind])
                df_words.drop(df_words.index[ind], axis=0, inplace=True)
                break
    kept_indices = df_words.index.tolist()
    return kept_indices
'''