import json
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer


topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


## Get data
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)
df_sel = df.filter(like='gpt4')

# Create empty lists to store the results
topic_indices = []
probs = []
for index, row in df_sel.iterrows():
    # Apply topic_model.transform to the element
    topic_index, prob = topic_model.transform(row)
    # Append the topic_index and prob values to the corresponding lists
    topic_indices.append(topic_index)
    probs.append(prob)
result_df = pd.DataFrame({'topic_index': topic_indices, 'prob': probs})
check_df = pd.concat([df['constraint'],result_df],axis=1)

# Show wordcloud
create_wordcloud(topic_model, topic=999)

topic_distr, _ = topic_model.approximate_distribution(row)

# Similarity of the topic distribution across paragraphs
R = []
for index, row in df_sel.iterrows():
    # Get topic distribution
    topic_distr, _ = topic_model.approximate_distribution(row)#4parasx2376
    r = cosine_similarity(topic_distr)
    indices = np.triu_indices(r.shape[0],k=1)
    r_vec = r[indices]
    R.append(r_vec)
    
plt.plot(topic_distr[2])

########################################################
## Concatenate the paragraphs for each item
df_sel['concatenated'] = df_sel.apply(lambda row: row['gpt4_p1'] + ' ' + row['gpt4_p2'] + ' ' + row['gpt4_p3'] + ' ' + row['gpt4_p4'], axis=1)

# get topic distribution of the concatenated text
topic_distr, _ = topic_model.approximate_distribution(df_sel['concatenated'])#Nitemx2376
plt.plot(topic_distr[2])

# calculate entropy
def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return entropy

entropies = np.apply_along_axis(calculate_entropy, axis=1, arr=topic_distr)
df['entropy'] = entropies

# plot the similarity values for each item
plt.figure(figsize=(10,6))
plt.bar(df.index, df['entropy'])
plt.xlabel('condition')
plt.ylabel('entropy')
plt.title('topic entropy value by condition')
plt.xticks(df.index, df['constraint'], rotation=45)
plt.tight_layout()
plt.savefig(r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/plots/Entropy_4.png')
plt.show()
