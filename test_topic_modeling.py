'''Get top distribution for each context'''
import json
import mne
import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
from bertopic import BERTopic

###############################
#get words in each story
my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
stories = ['story_lw1', 'story_the_black_willow', 'story_easy_money', 'story_cable_spool_fort']
fname = my_path + 'stimuli/text_with_wordlists/' + stories[0] + '.json'  
with open(fname,'r') as file:
    all_content = json.load(file)
words = [re.sub(r'[^\w\s]', '', word) for word in all_content if word.isalpha()]

# get cloze values for each word in all_content


#note: some words are missing in df_words compared to words


## meta data of words in subject 1
subject = str(1).zfill(2)
session = 0
task = 0
word_fname = my_path + f"/segments/words_sub{subject}_session{session}_task{task}.csv"
df_words = pd.read_csv(word_fname)

