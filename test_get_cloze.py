'''Get cloze values'''
import json
import mne
import pandas as pd
import numpy as np
import nltk
from wordfreq import zipf_frequency
import mne_rsa
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
import openai
import math
import numpy as np
import itertools

key_fname = r'C:\Users\lwang11\Dropbox (Partners HealthCare)\OngoingProjects\MASC-MEG\lab_api_key.txt'
#key_fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/lab_api_key.txt'
with open(key_fname,'r') as file:
    key = file.read()
openai.api_key = key #get it from your openai account


def get_sentences(all_content):
    '''Concert the words to sentences
    all_contents: list of words and punctuations'''
    sentences = []
    current_sentence = ""
    for word in all_content:
        current_sentence += word + " "
        if "." in word:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence:
        sentences.append(current_sentence.strip())
    return sentences



def get_completions(prompt):
    '''get the cloze values for every token in the input
    prompt: text input'''
    completions = openai.Completion.create(
        engine= "gpt-3.5-turbo",#"text-davinci-003",
        prompt=prompt,
        max_tokens=0,
        top_p=1,
        logprobs = 0,
        frequency_penalty=0,
        presence_penalty=0, 
        echo=True
    )    
    logprobs = completions["choices"][0]["logprobs"]["token_logprobs"]
    tokens = completions["choices"][0]["logprobs"]["tokens"]
    probs = [np.e ** prob if prob is not None else 1 for prob in logprobs]
    df = pd.DataFrame({'tokens':tokens,
                       'probs':probs})
    return df
    

def get_word_cloze(df,prompt):
    '''get the probability of every word in the input
    df: output of the model: 'tokens' and 'probs'
    prompt: text input used for calculating the token probability'''
    utterance = prompt.split()
    probs = []
    prob_list = []
    token_prob = 1
    placeholder = ""
    word_count = 0
    for i, token in enumerate(df["tokens"]):
        placeholder += token.strip()
        prob = df.loc[i, "probs"]
        if placeholder == utterance[word_count].strip():
            if prob_list:
                token_prob *= np.prod(prob_list)
            token_prob *= prob
            probs.append(token_prob)
            placeholder = ""
            word_count += 1
            prob_list = []
            token_prob = 1
        else:
            prob_list.append(prob)
    df_cloze = pd.DataFrame({'words':utterance,
                       'probs':probs})
    return df_cloze


def find_cloze_cw(df_cloze_all,df_words):
    '''match the cloze values with the cws identified in the MEG data'''
    cw_lists = df_words['word'].to_list()
    for i, word in enumerate(cw_lists):
        print(i)
        print(word)
        while df_cloze_all.loc[i, 'words'] != word:
            pop_ele = df_cloze_all.loc[i, 'words']
            print(f'pop out {pop_ele}')
            df_cloze_all = df_cloze_all.drop(i, axis=0)
            df_cloze_all.reset_index(drop=True, inplace=True)
    return df_cloze_all
        
##########################################################################
## combine parts for each story
my_path = r'C:\\Users\\lwang11\\Dropbox (Partners HealthCare)\\OngoingProjects\\MASC-MEG\\'

all_content = []
for itask in range(4):
    fname = my_path + 'stimuli/text_with_wordlists/lw1_produced_' + str(itask) + '.txt'
    with open(fname,'r') as file:
        content = file.read().split()
        #words = [re.sub(r'[^\w\s]', '', word) for word in content if word.isalpha()]
        all_content.extend(content)
fname = my_path + 'stimuli/text_with_wordlists/story_lw1.json'  
with open(fname,'w') as file:
    json.dump(all_content,file)
    

all_content = []
for itask in range(12):
    fname = my_path + 'stimuli/text_with_wordlists/the_black_willow_produced_' + str(itask) + '.txt'
    with open(fname,'r') as file:
        content = file.read().split()
        #words = [re.sub(r'[^\w\s]', '', word) for word in content if word.isalpha()]
        all_content.extend(content)
fname = my_path + 'stimuli/text_with_wordlists/story_the_black_willow.json'  
with open(fname,'w') as file:
    json.dump(all_content,file)


all_content = []
for itask in range(8):
    fname = my_path + 'stimuli/text_with_wordlists/easy_money_produced_' + str(itask) + '.txt'
    with open(fname,'r') as file:
        content = file.read().split()
        #words = [re.sub(r'[^\w\s]', '', word) for word in content if word.isalpha()]
        all_content.extend(content)
fname = my_path + 'stimuli/text_with_wordlists/story_easy_money.json'  
with open(fname,'w') as file:
    json.dump(all_content,file)
    

all_content = []
for itask in range(6):
    fname = my_path + 'stimuli/text_with_wordlists/cable_spool_fort_produced_' + str(itask) + '.txt'
    with open(fname,'r') as file:
        content = file.read().split()
        #words = [re.sub(r'[^\w\s]', '', word) for word in content if word.isalpha()]
        all_content.extend(content)
fname = my_path + 'stimuli/text_with_wordlists/story_cable_spool_fort.json'  
with open(fname,'w') as file:
    json.dump(all_content,file)
    
##############################################################
#get cloze value of words in each story
my_path = r'C:\\Users\\lwang11\\Dropbox (Partners HealthCare)\\OngoingProjects\\MASC-MEG\\'
stories = ['story_lw1', 'story_the_black_willow', 'story_easy_money', 'story_cable_spool_fort']

# -----------------------------------------------------------------
## Sentence context
# -----------------------------------------------------------------
i=1
fname = my_path + 'stimuli/text_with_wordlists/' + stories[i] + '.json'  
with open(fname,'r') as file:
    all_content = json.load(file)
    print(len(all_content))
#words = [re.sub(r'[^\w\s]', '', word) for word in all_content if word.isalpha()]

# get sentences
sentences = get_sentences(all_content)

# get cloze values for each word in a particular sentence
prompt=sentences[0]
df = get_completions(prompt)
df_cloze_s2 = get_word_cloze(df,prompt)

prompt = sentences[0] + ' ' + sentences[1]
df = get_completions(prompt)
df_cloze_two = get_word_cloze(df,prompt)

# -----------------------------------------------------------------
## Full context
# -----------------------------------------------------------------
i=1
fname = my_path + 'stimuli/text_with_wordlists/' + stories[i] + '.json'  
with open(fname,'r') as file:
    all_content = json.load(file)
    print(len(all_content))