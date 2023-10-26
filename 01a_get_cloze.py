'''Get cloze values'''
import json
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
import itertools
import os
from difflib import ndiff

key_fname = '/Users/lwang11/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/lab_api_key.txt'
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
        model="text-davinci-003",
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


def get_cloze_longcontext(all_content):
    '''In case the context is over 4099 tokens'''
    sentences = get_sentences(all_content.split())
    content_a = ' '.join(sentences[:150])
    content_b = ' '.join(sentences[100:])
    
    df = get_completions(content_a) #get cloze values for tokens
    df_cloze_a = get_word_cloze(df,content_a) #get cloze values for words
    
    df = get_completions(content_b) #get cloze values for tokens
    df_cloze_b = get_word_cloze(df,content_b) #get cloze values for words
    
    # Identify the overlapping parts
    n = 30
    list_a = df_cloze_a['words'].tolist()
    list_b = df_cloze_b['words'].tolist()
    found_match = False
    for i in range(len(list_a) - n + 1):
        if list_a[i:i+n] == list_b[:n]:
            found_match = True
            break
    
    overlap_a = list_a[i:len(list_a)]
    overlap_b = list_b[:len(overlap_a)]
    if overlap_a == overlap_b:
        df_b_cut = df_cloze_b[len(overlap_a):]
        df_cloze_all = pd.concat([df_cloze_a,df_b_cut],axis=0)
    
    return df_cloze_all

##########################################################################
#get words and cloze values in each story: the full precedding context
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
my_path = r'S:/USERS/Lin/MASC-MEG/'
stories = os.listdir(my_path + 'stimuli/text/')

for story in stories:
    fname = my_path + 'stimuli/text/' + story
    with open(fname,'r') as file:
        all_content = file.read()

    if len(all_content.split()) < 4000:
        df = get_completions(all_content) #get cloze values for tokens
        df_cloze_all = get_word_cloze(df,all_content) #get cloze values for words
    else:
        df_cloze_all = get_cloze_longcontext(all_content)
    
    df_fname = my_path + 'stimuli/cloze/cloze_FullContext_' + story.split('.')[0] + '.csv'
    df_cloze_all.to_csv(df_fname, index=False)

#df_cloze_all[df_cloze_all['probs'] > 0.9]['words'].tolist()

    
##########################################################################
#get words and cloze values in each story: each sentence as a context
#my_path = r'/Users/lwang11/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
my_path = r'S:/USERS/Lin/MASC-MEG/'
stories = os.listdir(my_path + 'stimuli/text/')

for story in stories:
    fname = my_path + 'stimuli/text/' + story
    with open(fname,'r') as file:
        all_content = file.read().split()
    #words = [re.sub(r'[^\w\s]', '', word) for word in all_content if word.isalpha()]

    # get sentences
    sentences = get_sentences(all_content)

    # get cloze values for each word in each sentence
    cloze_list = []
    for prompt in sentences:
        df = get_completions(prompt) #get cloze values for tokens
        df_cloze = get_word_cloze(df,prompt) #get cloze values for words
        cloze_list.append(df_cloze)

    df_cloze_all = pd.concat(cloze_list, ignore_index=True)
    df_fname = my_path + 'stimuli/cloze/cloze_SenContext_' + story.split('.')[0] + '.csv'
    df_cloze_all.to_csv(df_fname, index=False)

## Match words between the text data and the MEG metadata
#note: some words are missing in df_words compared to words
# see '01_generate_words_features.py'


'''
############################################################################################
# combine parts for each story: which contains both the story text and random word/sentence lists
# the subpart stories do not match with the full story text, OR the word list in MEG data
#my_path = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/MASC-MEG/'
my_path = r'S:/USERS/Lin/MASC-MEG/'

#----------------------
# story 1
# texts together with random word/sentence lists in the middle
all_content = []
for itask in range(4):
    fname = my_path + 'stimuli/text_with_wordlists/lw1_produced_' + str(itask) + '.txt'
    with open(fname,'r') as file:
        content = file.read().split()
        #words = [re.sub(r'[^\w\s]', '', word) for word in content if word.isalpha()]
        all_content.extend(content)

# only continuous texts
fname = my_path + 'stimuli/text/' + 'lw1.txt'
with open(fname,'r') as file:
    all_text = file.read().split()
            
# Find the differences between tokens
diff = list(ndiff(all_text, all_content))
extra_tokens_with_context = []
missing_tokens_with_context = []
context_window = 5  # Number of words around the extra token
for i, token in enumerate(diff):
    if token.startswith('+ '):
        extra_token = token[2:]
        context_start = max(0, i - context_window)
        context_end = min(len(diff), i + context_window + 1)
        context = ' '.join([diff[j][2:] for j in range(context_start, context_end) if diff[j].startswith(('  ', '+ '))])
        extra_tokens_with_context.append((extra_token, context))
    if token.startswith('- '):
        extra_token = token[2:]
        context_start = max(0, i - context_window)
        context_end = min(len(diff), i + context_window + 1)
        context = ' '.join([diff[j][2:] for j in range(context_start, context_end) if diff[j].startswith(('  ', '+ '))])
        missing_tokens_with_context.append((extra_token, context))


print("Extra tokens in full_content with context:")
for extra_token, context in extra_tokens_with_context:
    print(f"Extra Token: {extra_token}")
    print(f"Context: {context}\n")



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
#get words in each story
my_path = r'C:\\Users\\lwang11\\Dropbox (Partners HealthCare)\\OngoingProjects\\MASC-MEG\\'

stories = ['story_lw1', 'story_the_black_willow', 'story_easy_money', 'story_cable_spool_fort']
i=1
fname = my_path + 'stimuli/text_with_wordlists/' + stories[i] + '.json'  
with open(fname,'r') as file:
    all_content = json.load(file)
    print(len(all_content))
#words = [re.sub(r'[^\w\s]', '', word) for word in all_content if word.isalpha()]

# get sentences
sentences = get_sentences(all_content)
'''