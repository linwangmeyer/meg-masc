import json
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

######################################################################
####### Infopos stimuli #######
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)

def get_sentence_similarity(df):
    '''Get the pairwise similarity for all stimuli of one item
    df: dataframe series containing text stimuli'''  
    df.tolist()
    sen_emb = model.encode(df) #Nparagraph x vectorDimension
    r = cosine_similarity(sen_emb)
    indices = np.triu_indices(r.shape[0],k=1)
    r_vec = r[indices]
    return r_vec

Rs = []
for index, row in df.iterrows():
    r_vec = get_sentence_similarity(row.filter(like='gpt4'))
    Rs.append(r_vec)

# average pairwise similarity for each item
result = np.mean(np.array(Rs),axis=1)
df['mean_similarity'] = result

# plot the similarity values for each item
plt.figure(figsize=(10,6))
plt.bar(df.index, df['mean_similarity'])
plt.xlabel('condition')
plt.ylabel('mean similarity')
plt.title('mean simialrity value by condition')
plt.xticks(df.index, df['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/plots/SenSim_4.png')
plt.show()

